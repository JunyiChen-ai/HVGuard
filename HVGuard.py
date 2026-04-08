import datetime
import argparse
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import json
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
import random
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")


device = 'cuda' if torch.cuda.is_available() else 'cpu'
config_path = 'config.jsonl'
trained_model_path = f'./saved_models/best_model.pth'


class VideoDataset(Dataset):
    def __init__(self, label_mapping, text_path=None, audio_path=None, frame_path=None, mix_path=None, label_path=None, use_modalities=None, device='cuda'):
        """
        Dataset loader that reads and aligns the text, audio, and frame embeddings of a video,
        and loads the labels from the data.json file.

        Args:
            text_path (str): Path to the pth file containing text embeddings
            audio_path (str): Path to the pth file containing audio embeddings
            frame_path (str): Path to the pth file containing frame embeddings
            mix_path (str): Path to the pth file containing fused information embeddings
            label_path (str): Path to the JSON file containing labels
            use_modalities (list): Modalities to be used, e.g., ['text', 'audio', 'frames']
            device (str): Device to use, either 'cuda' or 'cpu'
        """

        self.device = device
        self.use_modalities = use_modalities if use_modalities else [
            'text', 'audio', 'frames']

        data_files = [text_path, audio_path, frame_path, mix_path, label_path]

        for file in tqdm(data_files, desc="Loading Datasets", unit="file"):
            # Load pth file (embedding cache)
            self.text_data = torch.load(
                text_path, map_location=device) if 'text' in self.use_modalities else None
            self.audio_data = torch.load(
                audio_path, map_location=device) if 'audio' in self.use_modalities else None
            self.frame_data = torch.load(
                frame_path, map_location=device) if 'frames' in self.use_modalities else None
            self.mix_data = torch.load(
                mix_path, map_location=device) if 'mix' in self.use_modalities else None

            self.label_mapping = label_mapping
            # Load label file
            with open(label_path, 'r', encoding="utf-8") as f:
                self.label_data = json.load(f)

        if isinstance(self.label_data, list):
            try:
                self.label_data = {item['Video_ID']: item for item in self.label_data}
            except KeyError:
                raise ValueError("data.json 中缺少 'Video_ID' 字段，请检查 JSON 文件的格式")

        self.video_ids = self._get_common_video_ids()

    def _get_common_video_ids(self):
        video_ids = set(self.label_data.keys())
        if self.text_data:
            video_ids &= set(self.text_data.keys())
        if self.audio_data:
            video_ids &= set(self.audio_data.keys())
        if self.frame_data:
            video_ids &= set(self.frame_data.keys())
        if self.mix_data:
            video_ids &= set(self.mix_data.keys())

        return sorted(video_ids)

    def __len__(self):
        return len(self.video_ids)

    def __getitem__(self, idx):
        video_id = self.video_ids[idx]

        embeddings = []
        if 'text' in self.use_modalities:
            embeddings.append(self.text_data[video_id])
        if 'audio' in self.use_modalities:
            embeddings.append(self.audio_data[video_id])
        if 'frames' in self.use_modalities:
            embeddings.append(self.frame_data[video_id])
        if 'mix' in self.use_modalities:
            embeddings.append(self.mix_data[video_id])

        embeddings = [e.to(self.device) for e in embeddings]
        combined_embedding = torch.cat(embeddings, dim=-1)
        label = self.label_data[video_id]['Label']
        label_int = self.label_mapping.get(label, -1)
        label_tensor = torch.tensor(label_int).to(device)
        return combined_embedding, label_tensor


class MoE(nn.Module):
    def __init__(self, input_dim, num_experts=4, expert_dim=128, output_dim=128):
        """
        Mixture of Experts (MoE).

        Args:
            input_dim (int): Dimension of the input features
            num_experts (int): Number of experts
            expert_dim (int): Hidden layer dimension of each expert network
            output_dim (int): Dimension of the output
        """

        super(MoE, self).__init__()
        """Create expert networks, each consisting of two linear layers and a ReLU activation function."""
        self.experts = nn.ModuleList([nn.Sequential(
            nn.Linear(input_dim, expert_dim),
            nn.ReLU(),
            nn.Linear(expert_dim, output_dim)
        ) for _ in range(num_experts)])
        self.gate = nn.Sequential(
            nn.Linear(input_dim, num_experts),
            nn.Softmax(dim=-1)
        )
        self.gate_dropout = nn.Dropout(0.1)

    def forward(self, x):
        gate_weights = self.gate(x)
        gate_weights = self.gate_dropout(gate_weights)
        expert_outputs = torch.stack([expert(x)
                                     for expert in self.experts], dim=1)
        output = torch.sum(gate_weights.unsqueeze(-1) * expert_outputs, dim=1)
        return output


class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_classes=10):
        """
        A standard MLP classifier employed as a baseline for comparison experiments.

        Args:
            input_dim (int): Dimension of the input features
            hidden_dim (int): Dimension of the hidden layer
            num_classes (int): Number of classes
        """

        super(MLPClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        return self.model(x)


def train(model, dataloader, criterion, optimizer, device='cpu'):
    """
    Function for training the model.

    Args:
        model (nn.Module): The model to be trained
        dataloader (DataLoader): Data loader for the training data
        criterion (nn.Module): Loss function
        optimizer (torch.optim.Optimizer): Optimizer
        device (str): Device to use, either 'cpu' or 'gpu'
    """

    model.train()
    total_loss = 0
    for embeddings, labels in dataloader:
        embeddings, labels = embeddings.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(embeddings)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)


def validate(model, dataloader, criterion, num_classes, device='cpu'):
    """
    Function for validating the model. Supports both three-class and binary classification, 
    automatically selecting the mode based on the number of labels.  

    Metrics computed include: Accuracy, Macro-F1, as well as Precision, Recall, and F1-score 
    for the Hateful (not applicable in binary classification) and Offensive labels.  

    Args:
        model (nn.Module): The model to be validated
        dataloader (DataLoader): Data loader for the validation data
        criterion (nn.Module): Loss function
        num_classes (int): Number of classes
        device (str): Device to use, either 'cpu' or 'gpu'
    """

    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []  
    all_labels = [] 

    label2_precision = 0
    label2_recall = 0
    label2_f1 = 0
    label1_precision = 0
    label1_recall = 0
    label1_f1 = 0
    label0_precision = 0
    label0_recall = 0
    label0_f1 = 0

    with torch.no_grad():
        for embeddings, labels in dataloader:
            embeddings, labels = embeddings.to(device), labels.to(device)
            outputs = model(embeddings)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            all_preds.extend(predicted.cpu().numpy())  
            all_labels.extend(labels.cpu().numpy())  

            if num_classes == 3:
                label2_preds = predicted == 2
                label1_preds = predicted == 1
                label0_preds = predicted == 0
                label2_true = labels == 2
                label1_true = labels == 1
                label0_true = labels == 0

                label2_precision += precision_score(
                    label2_true.cpu(), label2_preds.cpu(), zero_division=0)
                label2_recall += recall_score(label2_true.cpu(),
                                              label2_preds.cpu(), zero_division=0)
                label2_f1 += f1_score(label2_true.cpu(),
                                      label2_preds.cpu(), zero_division=0)

                label1_precision += precision_score(
                    label1_true.cpu(), label1_preds.cpu(), zero_division=0)
                label1_recall += recall_score(label1_true.cpu(),
                                              label1_preds.cpu(), zero_division=0)
                label1_f1 += f1_score(label1_true.cpu(),
                                      label1_preds.cpu(), zero_division=0)

                label0_precision += precision_score(
                    label0_true.cpu(), label0_preds.cpu(), zero_division=0)
                label0_recall += recall_score(label0_true.cpu(),
                                              label0_preds.cpu(), zero_division=0)
                label0_f1 += f1_score(label0_true.cpu(),
                                      label0_preds.cpu(), zero_division=0)

            elif num_classes == 2:
                label2_preds = predicted == 1
                label1_preds = predicted == 0
                label2_true = labels == 1
                label1_true = labels == 0

                label2_precision += precision_score(
                    label2_true.cpu(), label2_preds.cpu(), zero_division=0)
                label2_recall += recall_score(label2_true.cpu(),
                                              label2_preds.cpu(), zero_division=0)
                label2_f1 += f1_score(label2_true.cpu(),
                                      label2_preds.cpu(), zero_division=0)

                label1_precision += precision_score(
                    label1_true.cpu(), label1_preds.cpu(), zero_division=0)
                label1_recall += recall_score(label1_true.cpu(),
                                              label1_preds.cpu(), zero_division=0)
                label1_f1 += f1_score(label1_true.cpu(),
                                      label1_preds.cpu(), zero_division=0)

    accuracy = correct / total
    precision = precision_score(
        all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds,
                          average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    m_f1 = (label2_f1/len(dataloader) + label1_f1/len(dataloader) +
            label0_f1/len(dataloader)) / num_classes

    return (total_loss / len(dataloader), accuracy, precision, recall, f1, m_f1,
            label2_recall / len(dataloader), label2_precision /
            len(dataloader), label2_f1 / len(dataloader),
            label1_recall / len(dataloader), label1_precision / len(dataloader), label1_f1 / len(dataloader), label0_recall / len(dataloader), label0_precision / len(dataloader), label0_f1 / len(dataloader))


def evaluate(model, criterion, trained_model_path, dataset, video_ids, best_state, num_classes, split_ids=None):
    """
    Evaluate the model's performance on the validation and test sets, and report the evaluation results.

    Args:
        model (torch.nn.Module): The model to be evaluated.
        criterion (torch.nn.Module): Loss function.
        trained_model_path (str): Path to the trained model.
        dataset (torch.utils.data.Dataset): Dataset.
        video_ids (list): List of video IDs.
        best_state (int): Best state indicator.
        num_classes (int): Number of classes.
    """

    model.load_state_dict(torch.load(trained_model_path))
    model.to(device)
    model.eval() 
    if split_ids is not None:
        available_ids = set(video_ids)
        test_ids = [vid for vid in split_ids['test'] if vid in available_ids]
    else:
        test_ids, temp_ids = train_test_split(
            video_ids, test_size=0.8, random_state=best_state)
        val_ids, train_ids = train_test_split(
            temp_ids, test_size=0.875, random_state=best_state)
    test_dataset = torch.utils.data.Subset(
        dataset, [dataset.video_ids.index(id) for id in test_ids])
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    test_loss, test_accuracy, test_precision, test_recall, test_f1, test_m_f1, test_2_recall, test_2_precision, test_2_F1, test_1_recall, test_1_precision, test_1_F1, test_0_recall, test_0_precision, test_0_F1 = validate(
        model, test_dataloader, criterion, num_classes, device)

    if num_classes == 3:
        print(f"Accuracy: {test_accuracy:.4f}, "
              f"M-F1: {test_m_f1:.4f},"
              f"F1(H): {test_2_F1:.4f},"
              f"R(H): {test_2_recall:.4f}, "
              f"P(H): {test_2_precision:.4f}, "
              f"F1(O): {test_1_F1:.4f}, "
              f"R(O): {test_1_recall:.4f}, "
              f"P(O): {test_1_precision:.4f}, "
              f"F1(N): {test_0_F1:.4f}, "
              f"R(N): {test_0_recall:.4f}, "
              f"P(N): {test_0_precision:.4f}")

    elif num_classes == 2:
        print(f"Accuracy: {test_accuracy:.4f}, "
              f"M-F1: {test_m_f1:.4f},"
              f"F1(O): {test_2_F1:.4f},"
              f"R(O): {test_2_recall:.4f}, "
              f"P(O): {test_2_precision:.4f}, "
              f"F1(N): {test_1_F1:.4f}, "
              f"R(N): {test_1_recall:.4f}, "
              f"P(N): {test_1_precision:.4f}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Training model...")
    parser.add_argument('--dataset_name', type=str, default='Multihateclip',
                        choices=['Multihateclip', 'HateMM'], help='Dataset name')
    parser.add_argument('--language', type=str, default='English',
                        choices=['Chinese', 'English'], help='Language of the dataset')
    parser.add_argument('--num_classes', type=int, default=2,
                        choices=[2, 3], help='Number of classes for classification')
    parser.add_argument('--mode', type=str, default='predict',
                        choices=['train', 'predict'], help='Training mode or prediction mode')
    parser.add_argument('--split_mode', type=str, default='random',
                        choices=['random', 'fixed'], help='Use random split or predefined split files')

    return parser.parse_args()


def load_config(config_path):
    config = []
    with open(config_path, 'r') as f:
        for line in f:
            try:
                config.append(json.loads(line.strip()))
            except json.JSONDecodeError as e:
                print(f"JSONDecodeError: {e}")
                print(f"Offending line: {line}")
    return config


def get_paths_and_best_state(config, dataset_name, language, num_classes):
    for entry in config:
        if (entry['dataset_name'] == dataset_name and
            entry['language'] == language and
                entry['num_classes'] == num_classes):
            return entry['paths'], entry['best_state']
    raise ValueError(
        f"Invalid combination of dataset_name, language, num_classes: {dataset_name}, {language}, {num_classes}")


def load_split_ids(split_dir):
    split_ids = {}
    for split_name in ['train', 'valid', 'test']:
        split_path = os.path.join(split_dir, f'{split_name}.csv')
        with open(split_path, 'r', encoding='utf-8') as f:
            split_ids[split_name] = [row[0] for row in csv.reader(f) if row]
    return split_ids


def main():
    args = parse_args()
    config = load_config(config_path)
    Dataset_name = [args.dataset_name]  # (Multihateclip, HateMM)
    Language = [args.language]  # (Chinese, English)
    num_classes = args.num_classes  # (2, 3)
    mode = args.mode  # (train, predict)
    use_modalities = ['text', 'audio', 'frames', 'mix'] 
    epoch = 20  
    learning_rate = 1e-4  
    batch_size = 32  

    paths, best_state = get_paths_and_best_state(
        config, args.dataset_name, args.language, args.num_classes)

    if Dataset_name[0] == 'Multihateclip':
        if num_classes == 3:
            label_mapping = {
                'Normal': 0,
                'Offensive': 1,
                'Hateful': 2,
            }
        elif num_classes == 2:
            label_mapping = {
                'Normal': 0,
                'Offensive': 1,
                'Hateful': 1,
            }
        else:
            print("Error: Invalid number of classes for dataset")
            return

    elif Dataset_name[0] == 'HateMM':
        if num_classes == 2:
            label_mapping = {
                'Non Hate': 0,
                'Hate': 1,
            }
        else:
            print("Error: Invalid number of classes for dataset")
            return

    text_path = paths['text_path']
    audio_path = paths['audio_path']
    frame_path = paths['frame_path']
    mix_path = paths['mix_path']
    label_path = paths['label_path']
    model_path = paths['model_path']

    dataset = VideoDataset(label_mapping, text_path, audio_path,
                           frame_path, mix_path, label_path, use_modalities=use_modalities)
    video_ids = dataset.video_ids

    # random_state = random.randint(0, 10000)
    random_state = best_state
    split_ids = None

    if args.split_mode == 'fixed':
        if Dataset_name[0] == 'HateMM':
            split_dir = './datasets/HateMM/splits'
        else:
            split_dir = f'./datasets/{Dataset_name[0]}/{Language[0]}/splits'
        split_ids = load_split_ids(split_dir)
        available_ids = set(video_ids)
        train_ids = [vid for vid in split_ids['train'] if vid in available_ids]
        val_ids = [vid for vid in split_ids['valid'] if vid in available_ids]
        test_ids = [vid for vid in split_ids['test'] if vid in available_ids]
        print(
            f"Using fixed split from {split_dir}: "
            f"train={len(train_ids)}, valid={len(val_ids)}, test={len(test_ids)}"
        )
    else:
        # Split the dataset (70% training, 20% test, 10% validation)
        test_ids, temp_ids = train_test_split(
            video_ids, test_size=0.8, random_state=random_state)
        val_ids, train_ids = train_test_split(
            temp_ids, test_size=0.875, random_state=random_state)

    train_dataset = torch.utils.data.Subset(
        dataset, [dataset.video_ids.index(id) for id in train_ids])
    val_dataset = torch.utils.data.Subset(
        dataset, [dataset.video_ids.index(id) for id in val_ids])
    test_dataset = torch.utils.data.Subset(
        dataset, [dataset.video_ids.index(id) for id in test_ids])

    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False)

    input_dim = sum([
        dataset.text_data[list(dataset.text_data.keys())[
            0]].shape[-1] if 'text' in use_modalities else 0,
        dataset.audio_data[list(dataset.audio_data.keys())[
            0]].shape[-1] if 'audio' in use_modalities else 0,
        dataset.frame_data[list(dataset.frame_data.keys())[
            0]].shape[-1] if 'frames' in use_modalities else 0,
        dataset.mix_data[list(dataset.mix_data.keys())[0]
                         ].shape[-1] if 'mix' in use_modalities else 0
    ])

    # Configure parameters for the Mixture of Experts (MoE) network
    num_experts = 8 # # Number of experts
    expert_dim = 128  # Hidden layer dimension of the expert network
    moe_output_dim = 128  # # Output layer dimension of the expert network
    MLP_hidden_dim = 64  # # Hidden layer dimension of the MLP classifier
    model = nn.Sequential(
        MoE(input_dim=input_dim, num_experts=num_experts,
            expert_dim=expert_dim, output_dim=moe_output_dim),
        MLPClassifier(moe_output_dim, MLP_hidden_dim, num_classes)
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    if mode == "train":
        best_val_acc_overall = -1
        best_acc = -1
        print("Starting Training...")
        for num_epochs in range(epoch):
            train_loss = train(model, train_dataloader,
                               criterion, optimizer, device)
            _, val_accuracy, _, _, _, _, _, _, _, _, _, _, _, _, _, = validate(
                model, val_dataloader, criterion, num_classes, device)
            _, test_accuracy, _, _, _, _, _, _, _, _, _, _, _, _, _, = validate(
                model, test_dataloader, criterion, num_classes, device)

            print(
                f"Epoch {epoch}/{num_epochs+1}: Val Accuracy = {val_accuracy:.4f}")
            if val_accuracy > best_val_acc_overall:
                best_val_acc_overall = val_accuracy
                best_acc = test_accuracy
                print("This is a better acc.")
                torch.save(model.state_dict(), trained_model_path)

        print("Finished!")
        model.load_state_dict(torch.load(trained_model_path))
        best_model = f'./saved_models/{Dataset_name[0]}_{Language[0]}_{num_classes}_{best_acc}_{random_state}.pth'
        torch.save(model.state_dict(), best_model)
        print(
            f"New best model {Dataset_name[0]}_{Language[0]}_{num_classes}_{best_acc:.4f}_{random_state}.pth saved with test_accuracy {best_acc:.4f} (random_state={random_state})")
    else:
        best_model = model_path
    evaluate(model, criterion, best_model, dataset,
             video_ids, random_state, num_classes, split_ids=split_ids)


if __name__ == '__main__':
    main()
