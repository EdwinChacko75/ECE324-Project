import torch
from tqdm import tqdm
from dataset import extract_final_number, compute_accuracy
from utils import save_outputs_to_file


def train_step(model, tokenizer, batch, precision, optimizer, scheduler=None):
    prompts = batch["prompts"]
    ground_truth_values = batch["ground_truth_values"]

    # Tokenize inputs
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    ).to("cuda")

    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    # Create labels with padding masked out
    labels = input_ids.clone()
    labels[input_ids == tokenizer.pad_token_id] = -100

    optimizer.zero_grad()

    with torch.autocast("cuda", dtype=precision):
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        loss = outputs.loss

    loss.backward()
    optimizer.step()
    if scheduler:
        scheduler.step()

    return loss.item(), prompts, ground_truth_values

def train_model(
    model,
    tokenizer,
    train_loader,
    precision,
    epochs,
    output_file,
    optimizer,
    scheduler=None,
):
    model.train()

    for epoch in range(epochs):
        print(f"\nStarting epoch {epoch + 1}/{epochs}\n")
        accuracies = []
        total_loss = 0.0
        accuracy = 0

        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):
            loss, prompts, ground_truth_values = train_step(
                model, tokenizer, batch, precision, optimizer, scheduler
            )
            total_loss += loss

            # Generate outputs for accuracy eval
            with torch.no_grad():
                inputs = tokenizer(
                    prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512,
                ).to("cuda")

                generated_ids = model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_length=1000,
                )

            generated_texts = [
                tokenizer.decode(output, skip_special_tokens=True) for output in generated_ids
            ]

            batch_predictions = [extract_final_number(text) for text in generated_texts]
            batch_acc = compute_accuracy(batch_predictions, ground_truth_values)

            accuracies.append(batch_acc)
            accuracy = (accuracy * batch_idx + batch_acc) / (batch_idx + 1)

            print(f"Batch {batch_idx+1} - Loss: {loss:.4f} | Accuracy: {accuracy*100:.2f}%")

            save_outputs_to_file(
                output_file,
                batch_idx,
                prompts,
                generated_texts,
                ground_truth_values,
                batch_acc,
                accuracy,
            )

        avg_loss = total_loss / len(train_loader)
        avg_accuracy = sum(accuracies) / len(accuracies)

        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"   Avg Loss:     {avg_loss:.4f}")
        print(f"   Avg Accuracy: {avg_accuracy * 100:.2f}%\n")


def test_model(model, tokenizer, test_loader, prescision):
    model.eval()
    accuracies = []
    accuracy = 0
    # Inference loop
    for batch_idx, batch in enumerate(tqdm(test_loader, desc="Processing Batches")):
        prompts = batch["prompts"]
        batch_ground_truth_values = batch["ground_truth_values"]

        inputs = tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True
        ).to("cuda")


        with torch.no_grad():
            with torch.autocast("cuda", dtype=prescision):
                outputs = model.generate(
                    **inputs, max_length=1000
                )  # Can adjust as needed

        generated_texts = [
            tokenizer.decode(output, skip_special_tokens=True) for output in outputs
        ]

        batch_predictions = [extract_final_number(text) for text in generated_texts]

        batch_acc = compute_accuracy(batch_predictions, batch_ground_truth_values)
        accuracies.append(batch_acc)
        accuracy = (accuracy * batch_idx + batch_acc) / ((batch_idx + 1))

        # TODO: Put this on the tqdm bar
        print(f"Test Accuracy: {accuracy * 100:.2f}%")


    print(f"Model Accuracy: {accuracy * 100:.2f}%")
    print(f"Model Accuracy: {sum(accuracies)/len(accuracies) * 100:.2f}%")