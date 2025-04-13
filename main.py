import torch
from docling_core.types.doc import DoclingDocument
from docling_core.types.doc.document import DocTagsDocument
from transformers import AutoProcessor, AutoModelForVision2Seq
from transformers.image_utils import load_image

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_image_url(image_url):
    image = load_image(image_url)
    return image


def init():
    print("Enter init().")

    processor = AutoProcessor.from_pretrained("ds4sd/SmolDocling-256M-preview")
    model = AutoModelForVision2Seq.from_pretrained(
        "ds4sd/SmolDocling-256M-preview",
        torch_dtype=torch.bfloat16,
        _attn_implementation="eager",  # for gpu not support flash attention
    ).to(DEVICE)
    return processor, model


def create_messages():
    print("Enter create_messages().")

    messages = [
        {
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": "your prompt"}],
        },
    ]

    return messages


def prepare_inputs(processor, messages, image):
    print("Enter prepare_inputs().")

    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt, images=[image], return_tensors="pt")
    inputs = inputs.to(DEVICE)
    return inputs


def generate_outputs(model, inputs, processor):
    print("Enter generate_outputs().")

    generated_ids = model.generate(**inputs, max_new_tokens=8192)
    prompt_length = inputs.input_ids.shape[1]
    trimmed_generated_ids = generated_ids[:, prompt_length:]
    doctags = processor.batch_decode(
        trimmed_generated_ids,
        skip_special_tokens=False,
    )[0].lstrip()
    return doctags


def populate_documents(doctags, image):
    print("Enter populate_documents().")

    doctags_doc = DocTagsDocument.from_doctags_and_image_pairs([doctags], [image])
    print(doctags)
    return doctags_doc


def create_docling_doc(doctags_doc):
    print("Enter create_docling_doc().")

    doc = DoclingDocument(name="Document")
    doc.load_from_doctags(doctags_doc)

    return doc


def main():
    print("Enter main().")

    image = load_image_url("simple-table.png")
    processor, model = init()
    messages = create_messages()
    inputs = prepare_inputs(processor, messages, image)
    doctags = generate_outputs(model, inputs, processor)
    doctags_doc = populate_documents(doctags, image)
    doc = create_docling_doc(doctags_doc)

    print(doc.export_to_markdown())


if __name__ == "__main__":
    main()
