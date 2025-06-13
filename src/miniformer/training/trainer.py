import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


class TransformerTrainer:
    """Trainer for transformer models"""
    
    def __init__(self, model, learning_rate=5e-5, batch_size=32, device=None):
        """
        Args:
            model: Transformer model
            learning_rate: Learning rate
            batch_size: Batch size
            device: Device to run on (None for auto-detection)
        """
        self.model = model
        self.batch_size = batch_size
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Setup optimizer and loss function
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.98), eps=1e-9)
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding
        
    def train(self, dataset, epochs=10, validation_dataset=None):
        """
        Train the model
        
        Args:
            dataset: Training dataset
            epochs: Number of epochs
            validation_dataset: Optional validation dataset
            
        Returns:
            history: Training history
        """
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True
        )
        
        if validation_dataset:
            val_dataloader = torch.utils.data.DataLoader(
                validation_dataset, batch_size=self.batch_size
            )
        
        history = {'train_loss': [], 'val_loss': []}
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            total_loss = 0
            
            for batch in tqdm(dataloader, desc=f'Epoch {epoch+1}/{epochs}'):
                input_ids, labels = batch
                input_ids = input_ids.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                self.optimizer.zero_grad()
                outputs = self.model(input_ids)
                
                # Calculate loss (reshape for cross entropy)
                loss = self.criterion(
                    outputs.contiguous().view(-1, outputs.size(-1)),
                    labels.contiguous().view(-1)
                )
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
            
            avg_train_loss = total_loss / len(dataloader)
            history['train_loss'].append(avg_train_loss)
            
            # Validation
            if validation_dataset:
                val_loss = self.evaluate(val_dataloader)
                history['val_loss'].append(val_loss)
                print(f'Epoch {epoch+1}/{epochs} - Train loss: {avg_train_loss:.4f} - Val loss: {val_loss:.4f}')
            else:
                print(f'Epoch {epoch+1}/{epochs} - Train loss: {avg_train_loss:.4f}')
                
        return history
    
    def evaluate(self, dataloader):
        """Evaluate the model"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids, labels = batch
                input_ids = input_ids.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(input_ids)
                loss = self.criterion(
                    outputs.contiguous().view(-1, outputs.size(-1)),
                    labels.contiguous().view(-1)
                )
                
                total_loss += loss.item()
                
        return total_loss / len(dataloader)
