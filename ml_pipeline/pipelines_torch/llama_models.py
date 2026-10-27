"""Frozen llama.cpp feature adapter."""

import logging

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger("ml_pipeline.pipelines_torch.models")


class LlamaCppClassifier:
    """
    Adapter-style classifier/regressor for llama.cpp models.
    Trains a dense layer on top of frozen llama.cpp outputs for classification or regression.
    """

    __module__ = "ml_pipeline.pipelines_torch.models"
    def __init__(self, gguf_path, num_classes=2, task_type='classification', device="cpu"):
        from llama_cpp import Llama
        self.llama = Llama(model_path=gguf_path, n_ctx=2048, logits_all=True)
        self.task_type = task_type
        # You must know the hidden size of the model (e.g., 8192 for 27B)
        self.hidden_size = self.llama.config["hidden_size"]
        import torch
        import torch.nn as nn

        if task_type == 'classification':
            self.classifier = nn.Linear(self.hidden_size, num_classes).to(device)
        else:  # regression
            self.classifier = nn.Linear(self.hidden_size, 1).to(device)
        self.device = device

    def _get_logits(self, prompt):
        import torch
        # Get llama.cpp output (logits for each token)
        output = self.llama(prompt)
        # Use the last token's logits as the representation
        last_logits = torch.tensor(output["logits"][-1]).to(self.device)
        return last_logits

    def fit(self, prompts, labels, epochs=3, lr=1e-3):
        """
        Fine-tune the classifier/regressor head on top of llama.cpp outputs.
        Only the classifier head is trained.
        """
        import torch
        import torch.optim as optim
        import torch.nn as nn
        self.classifier.train()
        optimizer = optim.Adam(self.classifier.parameters(), lr=lr)

        if self.task_type == 'classification':
            loss_fn = nn.CrossEntropyLoss()
        else:  # regression
            loss_fn = nn.MSELoss()

        for epoch in range(epochs):
            total_loss = 0
            for prompt, label in zip(prompts, labels):
                logits = self._get_logits(prompt)
                pred = self.classifier(logits.unsqueeze(0))

                if self.task_type == 'classification':
                    loss = loss_fn(pred, torch.tensor([label]).to(self.device))
                else:  # regression
                    loss = loss_fn(pred.squeeze(), torch.tensor([label], dtype=torch.float32).to(self.device))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            logger.info("Epoch %d loss: %.4f", epoch + 1, total_loss / len(prompts))

    def predict(self, prompts):
        """
        For classification: predict class indices.
        For regression: predict continuous values.
        """
        import torch
        self.classifier.eval()
        preds = []
        with torch.no_grad():
            for prompt in prompts:
                logits = self._get_logits(prompt)
                pred = self.classifier(logits.unsqueeze(0))
                if self.task_type == 'classification':
                    preds.append(pred.argmax(dim=-1).item())
                else:  # regression
                    preds.append(pred.squeeze().item())

        # Diagnostic stats
        preds_array = np.array(preds)
        if self.task_type == 'classification':
            unique_preds, counts = np.unique(preds_array, return_counts=True)
            logger.debug("[%s] Prediction distribution: %s", self.__class__.__name__, dict(zip(unique_preds, counts)))
        else:
            logger.debug("[%s] Predictions - min: %.4f, max: %.4f, mean: %.4f, std: %.4f", self.__class__.__name__, preds_array.min(), preds_array.max(), preds_array.mean(), preds_array.std())

        return preds

    def predict_proba(self, prompts):
        """
        Predict class probabilities for classification tasks only.
        """
        if self.task_type != 'classification':
            raise ValueError("predict_proba is only available for classification tasks")
        import torch
        self.classifier.eval()
        probs = []
        with torch.no_grad():
            for prompt in prompts:
                logits = self._get_logits(prompt)
                pred = self.classifier(logits.unsqueeze(0))
                prob = torch.softmax(pred, dim=-1).cpu().numpy()[0]
                probs.append(prob)

        # Diagnostic stats
        probs_array = np.array(probs)
        predicted_classes = np.argmax(probs_array, axis=1)
        unique_preds, counts = np.unique(predicted_classes, return_counts=True)
        logger.debug("[%s] predict_proba class distribution: %s", self.__class__.__name__, dict(zip(unique_preds, counts)))
        logger.debug("   Probability stats - min: %.4f, max: %.4f, mean: %.4f", probs_array.min(), probs_array.max(), probs_array.mean())

        return probs

