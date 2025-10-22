import torch
import json
import wandb

import lightning as L
import pytorch_lightning as pl
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
from evaluate_utils import exam_taker, basic_stats, exam_taker_blimp, fractioned_exam_taker, exact_match_evaluation
from dataset import prepare_blimp_data
from sklearn.mixture import GaussianMixture
from losses import focal_loss, clipped_loss, entropy_loss, focal_clipped_loss, find_intersection, get_probability_of_correct_class, lovasz_softmax_flat

class LitModel(pl.LightningModule):
    def __init__(self,
                 cml_model, 
                 learning_rate=1e-3, 
                 epochs=10, 
                 focal_gamma=None, 
                 focal_alpha=None, 
                 evaluate_KT=None, 
                 blimp_eval=None,
                 evaluate_hellaswag=None, 
                 tokenizer=None,
                 fractions=None,
                 stage=None, 
                 eval_delta=0,
                 loss_type="baseline",
                 test_mode=False,
                 clip_dataset=None,
                 task="wikitext",
                 **kwargs):
        self.save_hyperparameters()
        super().__init__()
        self.model = cml_model.train()

        self.lr = learning_rate
        self.epochs = epochs

        self.test_step_loss_outputs = []
        self.test_step_perplexity_outputs = []

        self.gamma = focal_gamma
        self.alpha = focal_alpha

        self.evaluate_KT = evaluate_KT
        self.blimp_eval = blimp_eval
        self.hellaswag_eval = evaluate_hellaswag

        self.loss_type = loss_type
        self.probs_wandb = []
    
        if self.evaluate_KT is not None:
            self.kt_dataset = json.load(open(evaluate_KT))
        if blimp_eval is not None:
            self.blimp_eval_data = prepare_blimp_data(dataset_name=blimp_eval)
        if clip_dataset is not None:
            self.clip_dataset = clip_dataset
    
        self.tokenizer = tokenizer
        self.fractions = fractions
        self.stage = stage
        self.eval_delta = eval_delta
        self.args = {**kwargs}
        self.test_mode = test_mode
        self.task = task

        self.total_questions = 0
        self.correct_answers = 0

    def forward(self, x):
        return self.model(x, labels=x)
    
    def training_step(self, batch, batch_idx):
        outputs = self(batch['input_ids'])
        if self.loss_type.startswith("clpfocal") and self.gamma is not None and self.alpha is not None:
            loss = focal_clipped_loss(outputs.logits, batch['input_ids'], alpha=self.alpha, gamma=self.gamma)
        if self.loss_type.startswith("focal") and self.gamma is not None and self.alpha is not None:
            loss = focal_loss(outputs.logits, batch['input_ids'], alpha=self.alpha, gamma=self.gamma)
        if self.loss_type.startswith("entropy") and self.gamma is not None and self.alpha is not None:
            loss = entropy_loss(outputs.logits, batch['input_ids'], alpha=self.alpha, gamma=self.gamma)
        if self.loss_type.startswith("clipped") and self.gamma is not None and self.alpha is not None:
            loss = clipped_loss(outputs.logits, batch['input_ids'], alpha=self.alpha, gamma=self.gamma)
        if self.loss_type.startswith("dynamic") and self.gamma is not None and self.alpha is not None:
            loss = clipped_loss(outputs.logits, batch['input_ids'], alpha=self.alpha, gamma=self.gamma)
        if self.loss_type == "baseline":
            loss = outputs.loss
        if self.loss_type.startswith("lovasz"):
            logits = outputs.logits
            labels = batch['labels']
            # Shift so that tokens < n predict n
            logits = logits[..., :-1, :].contiguous()
            labels = labels[..., 1:].contiguous()
            # Prepare labels for non-cross-entropy losses
            # Set the labels to -100 before inputs["labels_position_id"] for each sample
            labels_start = batch["labels_position_id"].min()
            # We take from the start of the labels to the end of the sequence (not including the last token)
            processed_labels = labels[:, labels_start - 1 : -1].clone()
            for i, pos in enumerate(batch["labels_position_id"]):
                processed_labels[i, : pos - labels_start] = -100
            processed_logits = logits[:, labels_start - 1 : -1].contiguous()

            processed_probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            loss = lovasz_softmax_flat(processed_probs.reshape(-1, logits.size(-1)),
                                       batch["labels"].reshape(-1),
                                       )
        perplexity = torch.exp(outputs.loss)
        self.log('train/perplexity', perplexity, sync_dist=True)
        self.log('train/loss', loss, sync_dist=True)
        return loss

    def configure_optimizers(self):
        if self.args.get('optimizer') == 'sgd':
            optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9)
            scheduler = optim.lr_scheduler.LinearLR(optimizer, total_iters=self.epochs)
            return [optimizer], [scheduler]

        elif self.args.get('optimizer') == 'adamw':
            optimizer = optim.AdamW(self.model.parameters(), lr=self.lr)
            return [optimizer]

        else:
            optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
            # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
            return [optimizer]#, [scheduler]
        # optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

    def on_train_start(self):
        self.on_train_epoch_end()

    def on_train_epoch_start(self):
        if self.loss_type.startswith("dynamic"):
            probs = []
            for batch in tqdm(self.clip_dataset):
                outputs = self(batch['input_ids'].to(self.device))
                logits = outputs.logits.detach().cpu()
                labels = batch['input_ids']
                prob = get_probability_of_correct_class(logits, labels)
                probs.append(prob.mean().item())

            probs = np.asarray(probs).reshape(-1, 1)
            # Dynamic clipping threshold with 2 Gaussians
            # gmm = GaussianMixture(n_components=2, random_state=420)
            # gmm.fit(probs)

            # means = gmm.means_.flatten()
            # stds = np.sqrt(gmm.covariances_).flatten()

            # mu1, sigma1 = means[0], stds[0]  # Mean and std of the first normal distribution
            # mu2, sigma2 = means[1], stds[1]  # Mean and std of the second normal distribution
            # # Updating the threshold for clipping
            # self.gamma = find_intersection(mu1, sigma1, mu2, sigma2)
            
            # Dynamic clipping threshold with 1 Gaussian
            mean, std = np.mean(probs), np.std(probs)
            self.gamma = mean + self.alpha * std

            self.log("dynamic_clip/gamma", self.gamma)
            self.log("dynamic_clip/mean", mean)
            self.log("dynamic_clip/std", std)
            # Drawing the threhold line on the histogram
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.hist(probs.flatten(), bins=30, alpha=0.7, edgecolor='black')
            ax.axvline(self.gamma, color='red', linestyle='dashed', linewidth=2, label=f'Cutoff = {self.gamma}')
            ax.set_xlabel("Value")
            ax.set_ylabel("Frequency")
            ax.set_title(f"Probs with Cutoff Line")
            ax.legend()
            self.logger.log_image("dynamic_clip/probs_hist", [wandb.Image(fig)], self.current_epoch)

    def on_train_epoch_end(self):
        if self.test_mode == False:
            if self.task == "hellaswag":
                # Hellaswag exact match
                return
            
            if self.evaluate_KT is not None:
                results = fractioned_exam_taker(model=self, 
                                                tokenizer=self.tokenizer, 
                                                exam=self.kt_dataset, 
                                                fractions=self.fractions, 
                                                stage=self.stage,
                                                delta=self.eval_delta,
                                                )
                self.log('KT_total/accuracy', results['total']['accuracy'], sync_dist=True)
                self.log('KT_total/confidence', results['total']['confidence'], sync_dist=True)

                self.log('KT_upto_stage/accuracy', results[f"until_stage_{self.stage}"]['accuracy'], sync_dist=True)
                self.log('KT_upto_stage/confidence', results[f"until_stage_{self.stage}"]['confidence'], sync_dist=True)
                for i in range(self.fractions):
                    self.log(f'KT_fraction_{i}/accuracy', results[f"fraction_{i}"]['accuracy'], sync_dist=True)
                    self.log(f'KT_fraction_{i}/confidence', results[f"fraction_{i}"]['confidence'], sync_dist=True)
                    
            if self.blimp_eval is not None:
                results = exam_taker_blimp(self, self.tokenizer, self.blimp_eval_data)
                accuracy, confidence = basic_stats(results)
                self.log('blimp/accuracy', accuracy, sync_dist=True)
                self.log('blimp/confidence', confidence, sync_dist=True)
    
    def validation_step(self, batch, batch_idx):
        if self.task == "hellaswag":
            # Hellaswag exact match
            acc = exact_match_evaluation(self.model, batch)
            # self.log('val/accuracy', acc, sync_dist=True)
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            true_labels = batch["labels_gt"]

            # Number of candidates per example in HellaSwag (usually 4)
            num_endings = 4
            num_samples = len(true_labels) // num_endings

            correct = 0
            total = num_samples

            # Evaluate in batches
            for i in range(0, len(input_ids), num_endings):
                # Get a batch of context + ending pairs
                batch_input_ids = input_ids[i : i + num_endings]
                batch_attention_mask = attention_mask[i : i + num_endings]

                # Get log-likelihood for each ending
                outputs = self(batch_input_ids)
                logits = outputs.logits

                # Compute log probability for each ending
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = batch_input_ids[:, 1:].contiguous()
                loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
                per_token_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                per_sample_loss = per_token_loss.view(batch_input_ids.size(0), -1).sum(dim=1)

                # Get index of the ending with the highest probability (lowest loss)
                predicted_label = torch.argmin(per_sample_loss).item()

                # Compare with ground truth label
                if predicted_label == true_labels[i // num_endings]:
                    correct += 1

            # Compute exact match accuracy
            self.total_questions += total
            self.correct_answers += correct
            exact_match_score = correct / total
            return exact_match_score
        else:
            outputs = self(batch['input_ids'])
            loss = outputs.loss
            perplexity = torch.exp(loss)
            self.log('val/perplexity', perplexity, sync_dist=True)
            self.log('val/loss', loss, sync_dist=True)
            return loss

    def on_validation_epoch_end(self):
        if self.task == "hellaswag":
            exact_match_score = self.correct_answers / self.total_questions
            self.log('val/accuracy', exact_match_score, sync_dist=True)
            self.total_questions = 0
            self.correct_answers = 0

