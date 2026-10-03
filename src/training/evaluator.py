from collections import Counter

import torch


class Evaluator:
    """Validation metrics for the Appendix B experiments.

    `evaluate` returns the paper's headline number: the proportion of
    validation classes for which the *entire* predicted tuple of integers is
    correct. `evaluate_per_coefficient` splits that by coefficient and adds a
    majority-class baseline for each, which is what Table 11 reports.

    Decoding is greedy throughout. Sampling from the softmax instead (the old
    behaviour of `Transformer.generate`) measures a random draw from the model
    and scores several points lower than the numbers in the paper.
    """

    def __init__(self, model, dataloader, criterion, device, tokenizer=None):
        self.model = model
        self.dataloader = dataloader
        self.criterion = criterion
        self.device = device
        self.tokenizer = tokenizer

    def _batch(self, inputs, targets, max_new_tokens):
        """Loss and greedy predictions for one batch.

        The returned predictions are aligned with ``targets[:, -max_new_tokens:]``.
        """
        if self.model.architecture == 'encoder_only':
            logits = self.model(src=inputs)
            preds = logits.argmax(dim=-1)
            loss = self.criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
        elif self.model.architecture == 'encoder_decoder':
            logits = self.model(src=inputs, tgt=targets[:, :-1])
            preds = self.model.generate(src=inputs, tgt=targets[:, :-max_new_tokens],
                                        max_new_tokens=max_new_tokens)
            # shifted loss: position i of the decoder output predicts targets[:, i+1]
            loss = self.criterion(logits.view(-1, logits.size(-1)),
                                  targets[:, 1:].contiguous().view(-1))
        else:  # decoder_only
            full_sequence = torch.cat([inputs, targets], dim=1)
            preds = self.model.generate(src=None, tgt=full_sequence[:, :-max_new_tokens],
                                        max_new_tokens=max_new_tokens)
            logits = self.model(tgt=full_sequence[:, :-1])
            pred_targets = logits[:, inputs.size(1) - 1:]
            loss = self.criterion(pred_targets.contiguous().view(-1, pred_targets.size(-1)),
                                  targets.view(-1))
        return loss, preds[:, -max_new_tokens:]

    def evaluate(self, max_new_tokens):
        """Return (mean validation loss, joint exact-match accuracy)."""
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        n_samples = len(self.dataloader.dataset)

        with torch.no_grad():
            for inputs, targets in self.dataloader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                loss, preds = self._batch(inputs, targets, max_new_tokens)
                total_loss += loss.item()
                gold = targets[:, -max_new_tokens:]
                total_correct += (preds == gold).all(dim=1).sum().item()

        return total_loss / len(self.dataloader), total_correct / n_samples

    def evaluate_per_coefficient(self, max_new_tokens, n_coefficients=None):
        """Per-coefficient accuracy and majority-class baselines (Table 11).

        Requires a tokenizer, to turn token ids back into integers. Classes
        whose target does not decode to `n_coefficients` integers are skipped;
        a prediction that decodes to too few integers simply counts as wrong.
        """
        if self.tokenizer is None:
            raise ValueError("evaluate_per_coefficient needs a tokenizer")

        self.model.eval()
        decode = self.tokenizer.detokenize
        n_seen = joint = 0
        correct, truth = None, None

        with torch.no_grad():
            for inputs, targets in self.dataloader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                _, preds = self._batch(inputs, targets, max_new_tokens)
                for pred_row, tgt_row in zip(preds.tolist(), targets.tolist()):
                    true = decode(tgt_row)
                    if n_coefficients is None:
                        n_coefficients = len(true)
                    if len(true) != n_coefficients:
                        continue
                    if correct is None:
                        correct = [0] * n_coefficients
                        truth = [Counter() for _ in range(n_coefficients)]
                    # the prediction covers only the generated tail, so decode it
                    # on its own rather than reusing the target's leading <BOS>
                    pred = decode(pred_row)
                    n_seen += 1
                    joint += pred == true
                    for i, (p, t) in enumerate(zip(pred, true)):
                        truth[i][t] += 1
                        correct[i] += p == t
                    for i in range(len(pred), n_coefficients):
                        truth[i][true[i]] += 1

        if not n_seen:
            raise ValueError("no validation targets decoded to a consistent length")

        return {
            'n': n_seen,
            'joint': joint / n_seen,
            'accuracy': [c / n_seen for c in correct],
            'baseline': [max(t.values()) / n_seen for t in truth],
        }
