import numpy as np

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from dataset import *
from scipy.stats import chi2
from tqdm import tqdm

def to_tokens_and_logprobs(model, tokenizer, input_texts):
    device = model.device
    input_ids = tokenizer(input_texts, padding=True, return_tensors="pt").input_ids.to(device)

    outputs = model(input_ids)
    probs = torch.log_softmax(outputs.logits, dim=-1).detach()

    # Collect the probability of the generated token -- probability at index 0 corresponds to the token at index 1
    probs = probs[:, :-1, :]
    input_ids = input_ids[:, 1:]
    gen_probs = torch.gather(probs, 2, input_ids[:, :, None]).squeeze(-1)
    # torch.gather takes pro

    batch = []
    for input_sentence, input_probs in zip(input_ids, gen_probs):
        text_sequence = []
        for token, p in zip(input_sentence, input_probs):
            if token not in tokenizer.all_special_ids:
                text_sequence.append((tokenizer.decode(token), p.item()))
        batch.append(text_sequence)
    return batch

def question_answerer(model, tokenizer, question, answers):
    input_texts = [question + " " + answer for answer in answers]
    answer_start_index = len(tokenizer.tokenize(question))
    answer_end_index = min(len(tokenizer.tokenize(input_texts[0])), len(tokenizer.tokenize(input_texts[1]))) # Why shorter answer? That's not necessarily the correct answer (true one)
    batch = to_tokens_and_logprobs(model, tokenizer, input_texts)
    scores = []
    for text_sequence in batch:
        # score = sum(p for _, p in text_sequence)
        score = sum(p for _, p in text_sequence[answer_start_index: answer_end_index]) / (answer_end_index - answer_start_index)
        scores.append(score)
    return scores

def exam_taker(model, tokenizer, exam):
    results = []
    for question in tqdm(exam, desc="Taking exam kt", leave=False):
        context = question["context"]
        sentence_true = question["sentence_true"]
        sentence_false = question["sentence_false"]
        answers = [sentence_true, sentence_false]
        scores = question_answerer(model, tokenizer, context, answers)
        results.append(scores)
    return np.array(results)

def basic_stats(scores, resample_size=1000):
    scores = scores[:,0] - scores[:,1]
    accuracies = []
    for _ in range(resample_size):
        sample = np.random.choice(scores, len(scores) // 2, replace=False)
        accuracy = np.sum(sample > 0) / len(sample)
        accuracies.append(accuracy)
    accuracy = np.mean(accuracies)
    confidence = 1.96 * np.std(accuracies)
    return accuracy, confidence

def pvalue_calculator(scores1, scores2):
    scores1 = scores1[:,0] - scores1[:,1] > 0
    scores2 = scores2[:,0] - scores2[:,1] > 0
    # a : number of instances both models are correct
    # b : number of instances model 1 is correct and model 2 is wrong
    # c : number of instances model 1 is wrong and model 2 is correct
    # d : number of instances both models are wrong
    a = np.sum(scores1 & scores2)
    b = np.sum(scores1 & ~scores2)
    c = np.sum(~scores1 & scores2)
    d = np.sum(~scores1 & ~scores2)

    x2 = (b - c)**2 / (b + c)
    p = 1 - chi2.cdf(x2, df=1)
    
    return p, x2

def question_answerer_blimp(model, tokenizer, sentesnces):
    input_texts = [answer for answer in sentesnces]
    answer_start_index = 0
    answer_end_index = min(len(tokenizer.tokenize(input_texts[0])), len(tokenizer.tokenize(input_texts[1])))
    batch = to_tokens_and_logprobs(model, tokenizer, input_texts)
    scores = []
    for text_sequence in batch:
        # score = sum(p for _, p in text_sequence)
        score = sum(p for _, p in text_sequence[answer_start_index: answer_end_index]) / (answer_end_index - answer_start_index)
        scores.append(score)
    return scores

def exam_taker_blimp(model, tokenizer, exam):
    results = []
    for question in tqdm(exam, desc="Taking exam blimp", leave=False):
        sentence_true = question["sentence_good"]
        sentence_false = question["sentence_bad"]
        answers = [sentence_true, sentence_false]
        scores = question_answerer_blimp(model, tokenizer, answers)
        results.append(scores)
    return np.array(results)

def question_answerer_babi(model, tokenizer, question, answers):
    input_texts = [question + " " + answer for answer in answers]
    answer_start_index = len(tokenizer.tokenize(question))
    answer_end_index = len(tokenizer.tokenize(input_texts[0]))
    batch = to_tokens_and_logprobs(model, tokenizer, input_texts)
    scores = []
    for text_sequence in batch:
        # score = sum(p for _, p in text_sequence)
        score = sum(p for _, p in text_sequence[answer_start_index: answer_end_index]) / (answer_end_index - answer_start_index)
        scores.append(score)
    return scores

def exam_taker_babi(model, tokenizer, exam):
    results = []
    for question in tqdm(exam, desc="Taking exam kt", leave=False):
        context = question["context"]
        answers = [question["answer_correct"], question["answer_wrong"]]
        scores = question_answerer_babi(model, tokenizer, context, answers)
        results.append(scores)
    return np.array(results)

def fractioned_exam_taker(model, tokenizer, exam, fractions, stage, delta=0):
    raw_results = exam_taker(model, tokenizer, exam)
    results = {}
    total_exam_length = len(exam)
    eval_chunks = total_exam_length // fractions
    for i in range(fractions):
        accuracy, confidence = basic_stats(raw_results[i * eval_chunks + delta: (i + 1) * eval_chunks - delta])
        results["fraction_" + str(i)] = {"accuracy": accuracy, "confidence": confidence}
    
    accuracy, confidence = basic_stats(raw_results[:(stage+1) * eval_chunks])
    results["until_stage_" + str(stage)] = {"accuracy": accuracy, "confidence": confidence}

    accuracy, confidence = basic_stats(raw_results)
    results["total"] = {"accuracy": accuracy, "confidence": confidence}

    return results


def exact_match_evaluation(model, tokenized_data):
    """Perform exact match (EM) evaluation."""
    input_ids = tokenized_data["input_ids"]
    attention_mask = tokenized_data["attention_mask"]
    true_labels = tokenized_data["labels_gt"]

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
        outputs = model(batch_input_ids, attention_mask=batch_attention_mask)
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
    exact_match_score = correct / total
    return exact_match_score