"""Hugging Face fitting and resumable Trainer state for the QLoRA adapter."""

import logging

logger = logging.getLogger("ml_pipeline.pipelines_torch.models")


class HuggingFaceTrainingMixin:
    def fit(self, X, y, *args, **kwargs):
        """
        Fine-tune the model using QLoRA.
        X: list of texts, y: labels/targets
        Returns: training_history (list of dicts with metrics per epoch)
        """
        from transformers import TrainingArguments, Trainer, DataCollatorWithPadding, TrainerCallback
        from datasets import Dataset
        import torch
        import numpy as np
        from sklearn.preprocessing import LabelEncoder

        # Convert labels to proper format - handle string labels
        if isinstance(y, (list, np.ndarray)):
            if hasattr(y, 'to_numpy'):
                y = y.to_numpy()
            y = np.array(y).flatten()

            # Check if labels are strings and encode them
            if y.dtype == object or isinstance(y[0], str):
                label_encoder = LabelEncoder()
                y = label_encoder.fit_transform(y)
                # Store the encoder for later use in predict
                self.label_encoder = label_encoder
                logger.info("Label encoding: %s", dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))))

            y = y.astype(int).tolist()
        elif hasattr(y, 'tolist'):
            y = y.tolist()
            # Check if labels are strings
            if isinstance(y[0], str):
                label_encoder = LabelEncoder()
                y = label_encoder.fit_transform(y).tolist()
                self.label_encoder = label_encoder
                logger.info("Label encoding: %s", dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))))
            # Flatten if nested
            elif isinstance(y[0], (list, np.ndarray)):
                y = [int(item[0] if isinstance(item, (list, np.ndarray)) else item) for item in y]

        # Prepare dataset for sequence classification (not conversation format)
        dataset = Dataset.from_dict({'text': list(X), 'label': y})  # Use 'label' not 'labels' initially

        # Store max_seq_length for use in predict()
        self.max_seq_length = kwargs.get('max_seq_length', self.max_seq_length)

        # Tokenize the dataset - DataCollatorWithPadding expects 'label' not 'labels'
        def tokenize_fn(examples):
            # Tokenize the text
            tokenized = self.tokenizer(
                examples['text'],
                truncation=True,
                padding=False,  # Don't pad here, let DataCollator handle it
                max_length=self.max_seq_length
            )
            if self._needs_token_type_ids and 'token_type_ids' not in tokenized:
                tokenized['token_type_ids'] = [[0] * len(ids) for ids in tokenized['input_ids']]
            # Keep the label (DataCollator will rename to labels)
            tokenized['label'] = examples['label']
            return tokenized

        dataset = dataset.map(tokenize_fn, batched=True, remove_columns=['text'])
        # Don't set format yet - let the data collator handle it

        # Determine dtype for training based on GPU capability
        try:
            compute_capability = torch.cuda.get_device_capability()[0]
            use_bf16 = compute_capability >= 8
            use_fp16 = compute_capability < 8
        except Exception:
            # Fallback for older GPUs
            use_bf16 = False
            use_fp16 = True

        # Disable fp16/bf16 if using quantization to avoid gradient scaling issues
        # Quantized models already use mixed precision internally
        if self._is_quantized:
            use_bf16 = False
            use_fp16 = False

        # Training configuration - use standard TrainingArguments for sequence classification
        num_epochs = kwargs.get('epochs', 2)
        training_args = TrainingArguments(
            output_dir=kwargs.get('output_dir', './qlora_results'),
            num_train_epochs=num_epochs,
            per_device_train_batch_size=kwargs.get('batch_size', 1),
            gradient_accumulation_steps=kwargs.get('gradient_accumulation_steps', self._gradient_accumulation_steps),
            gradient_checkpointing=True,
            # use_reentrant=False is required for causal LMs and hybrid architectures
            # (Qwen3.5, Gemma 4, Llama) to avoid autograd graph issues
            gradient_checkpointing_kwargs={"use_reentrant": False},
            optim="adamw_torch_fused",
            logging_steps=10,
            learning_rate=kwargs.get('learning_rate', 1e-4),
            fp16=use_fp16,
            bf16=use_bf16,
            max_grad_norm=0.3,
            warmup_steps=0,
            lr_scheduler_type="constant",
            report_to="none",
            save_strategy="no",
            save_total_limit=2,
        )

        # Create data collator for dynamic padding
        data_collator = DataCollatorWithPadding(
            tokenizer=self.tokenizer,
            padding=True,
            return_tensors='pt'
        )

        # Create a custom callback to track training history
        training_history = []

        class MetricsCallback(TrainerCallback):
            def on_epoch_end(self, args, state, control, **callback_kwargs):
                if state.epoch is None or not np.isclose(state.epoch, round(state.epoch)):
                    return control
                # Get the latest log entry (contains loss)
                if state.log_history:
                    latest_log = state.log_history[-1]
                    epoch_metrics = {
                        'epoch': state.epoch,
                        'loss': latest_log.get('loss', 0.0),
                    }
                    training_history.append(epoch_metrics)
                    logger.info("Epoch %d/%d - Loss: %.4f", int(state.epoch), num_epochs, epoch_metrics['loss'])

        # Create trainer with standard Trainer (not SFTTrainer)
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
            processing_class=self.tokenizer,  # Use processing_class instead of tokenizer
            data_collator=data_collator,
            callbacks=[MetricsCallback()],
        )

        from pathlib import Path
        from ml_pipeline.utils.training_checkpoint import (
            CheckpointClock, CheckpointPause, StopRequest, atomic_save,
            atomic_transformers_save, load_checkpoint, restore_rng, rng_state,
        )
        output_dir = Path(training_args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        context = {"model_name": getattr(self.model.config, "_name_or_path", type(self.model).__name__), "epochs": num_epochs,
                   "batch_size": training_args.per_device_train_batch_size,
                   "gradient_accumulation_steps": training_args.gradient_accumulation_steps,
                   "learning_rate": training_args.learning_rate, "max_seq_length": self.max_seq_length,
                   "texts": list(X), "labels": y}
        context_file = output_dir / "restart_context.pt"
        previous = load_checkpoint(context_file, context)
        if previous is None:
            atomic_save({"config": context}, context_file)
        native_save = trainer._save_checkpoint
        clock = CheckpointClock()

        def save_checkpoint(model, trial=None):
            atomic_transformers_save(trainer, native_save, model, trial, {
                "history": training_history,
            })
            clock.saved()

        trainer._save_checkpoint = save_checkpoint
        checkpoints = sorted(
            (path for path in output_dir.glob("checkpoint-*")
             if path.is_dir() and (path / "restart_complete.pt").exists()),
            key=lambda path: int(path.name.split("-")[-1]),
        )
        resume = str(checkpoints[-1]) if checkpoints else None
        if resume:
            training_history.extend(load_checkpoint(Path(resume) / "restart_complete.pt")["history"])
        if previous is not None and previous.get("completed"):
            if resume is None:
                raise ValueError("Completed QLoRA restart has no native trainer checkpoint")
            trainer._load_from_checkpoint(resume, self.model)
            restore_rng(previous["rng"])
            return training_history
        with StopRequest() as stop_request:
            class RestartCallback(TrainerCallback):
                def on_train_begin(self, args, state, control, **callback_kwargs):
                    if resume is None:
                        trainer._save_checkpoint(trainer.model, trial=None)
                    return control

                def on_step_end(self, args, state, control, **callback_kwargs):
                    if clock.due() or stop_request.requested:
                        control.should_save = True
                    if stop_request.requested:
                        control.should_training_stop = True
                    return control

            trainer.add_callback(RestartCallback())
            trainer.train(resume_from_checkpoint=resume)
            trainer._save_checkpoint(trainer.model, trial=None)
            if stop_request.requested:
                raise CheckpointPause("QLoRA optimizer, scheduler, RNG and sampler progress saved")
        atomic_save({"config": context, "completed": True, "rng": rng_state()}, context_file)
        return training_history

