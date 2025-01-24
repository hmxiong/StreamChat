import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
torch.random.manual_seed(0)

def test_phi():
    model = AutoModelForCausalLM.from_pretrained(
        "/13390024681/All_Model_Zoo/Phi-3-mini-128k-instruct", 
        device_map="cuda", 
        torch_dtype="auto", 
        trust_remote_code=True, 
    )
    tokenizer = AutoTokenizer.from_pretrained("/13390024681/All_Model_Zoo/Phi-3-mini-128k-instruct")

    time_1 = time.time()
    messages = [
        {"role": "system", "content": "Your are a good summarizer."},
        {"role": "user", "content": "Help me summarize the following message: 1. The video shows the surrounding environment as a lawn on which a man is standing, holding a green bucket. 2. The video shows a man holding a white wooden board and using an electronic cutting machine to adjust the dimensions of the board."},
    ]

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
    )

    generation_args = {
        "max_new_tokens": 256,
        "return_full_text": False,
        "temperature": 0.3,
        "do_sample": False,
    }
    time_2 = time.time()

    output = pipe(messages, **generation_args)

    time_3 = time.time()

    print(output[0]['generated_text'])

    print("time spend:{}/{}".format((time_2 - time_1), (time_3 - time_2)))
    
def test_longva_only_text():
    from longva.utils import disable_torch_init
    from longva.conversation import conv_templates, SeparatorStyle
    from longva.model.builder import load_pretrained_model
    
    model_path = "/13390024681/All_Model_Zoo/LongVA-7B-DPO"
    disable_torch_init()
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, None, "llava_qwen")
    
    print("Model Initializ finish !!! ")
    
    caption_list = [
        "The video appears to be a first-person perspective showing an individual engaged in some sort of outdoor project or task. The person is standing on what looks like a raised platform or deck, with a clear view of the surrounding yard and driveway below. \
        In the foreground, there's a large, cylindrical object that resembles a barrel or container, possibly made of plastic or a similar material. It has a dark green or teal color and is quite prominent in the scene. The individual is holding what seems to be a long, thin piece of wood or a similar material, which they are either installing or preparing to install. \
        The wood is white and appears to be of a consistent thickness, suggesting it could be used for construction or landscaping purposes. The environment includes well-maintained grass, a driveway leading to a house or building, and various landscaping elements such as plants and garden ornaments. \
        There is also a glimpse of a structure that might be part of a shed or storage unit, and a vehicle parked at the edge of the driveway.The person is dressed in casual outdoor attire suitable for yard work or home improvement tasks, and their hands are visible as they manipulate the object they're working with. The overall setting suggests a suburban or residential area.",
        "This is a video showing a step-by-step process of constructing or repairing a large, flat object, likely a piece of outdoor furniture or a structure. The environment appears to be an outdoor setting with grass and some garden elements in the background. There are various tools and equipment visible on a workbench or table, including what looks like a router or sanding machine, clamps, and measuring tools. \
        In the foreground, there's a person who seems to be the main operator, working on the object. They are wearing casual clothing suitable for outdoor work, including a long-sleeved shirt with a logo on it, jeans, and sneakers. The person is focused on their task, handling the object carefully and using the tools to shape or finish it. \
        The video has a first-person perspective, as if you're looking down upon the scene from above or slightly elevated, giving a clear view of the entire process without any obstructions. The quality of the video suggests it might have been taken during a DIY project or workshop, possibly shared on social media or a tutorial platform.",
        "This is a video showing a person engaged in woodworking or construction work outdoors. The individual appears to be using specialized tools and equipment, possibly for shaping or cutting materials like wood or plastic. The environment suggests a residential area with grassy lawn and steps leading up to a house or porch. There are various tools and materials visible on a portable workbench or stand, including power drills, saws, and other hand tools. \
        The person is dressed in safety gear suitable for the task at hand, such as an orange safety vest and closed-toe shoes. The focus of the video is on the hands and tools, providing a close-up view of the work being done.",
        "The video captures a room in the midst of renovation or construction. The floor is covered with various tools and materials, including what appears to be wooden planks or boards, suggesting that flooring work is either underway or has been recently completed. There are also rolls of cable or wire running across the floor, possibly for electrical or data wiring. \
        In the center of the room, there's a person who seems to be involved in the work. They are wearing protective gear such as safety goggles and gloves, indicating they are taking precautions during the renovation process. The person is holding a long, straight object that could be a piece of wood or a tool used in the construction process. \
        The walls are painted in a light color, and the ceiling is visible, showing it may be unfinished or undergoing work. There are no visible texts or distinctive brands within the frame that provide additional context about the location or the nature of the renovation. The overall impression is one of an active home improvement project.    ",
        "This is a video showing a person in the midst of home renovation or construction work. The individual appears to be kneeling on a wooden floor, holding a long, straight-edged tool that could be used for measuring or leveling purposes. They are wearing an orange safety vest and blue work pants, which suggests they are taking precautions typical of such jobs. \
        The surrounding environment is a residential interior space undergoing transformation. There's visible construction debris and tools scattered around, indicating ongoing work. A variety of items like power drills, extension cords, and other equipment are present, further emphasizing the active nature of the project. The room has a neutral color palette with light-colored walls and flooring, and there are curtains partially drawn at the windows, suggesting privacy or perhaps a temporary setup during the renovation process."
    ]
    
    def make_summary_prompt(caption_list):
        order = [
                "first",
                "second",
                "third",
                "fourth",
                "fifth",
                "sixth",
                "seventh",
                "eighth",
                "ninth",
                "tenth"
            ]
        
        new_caption =[]
        for index, caption in enumerate(caption_list):
            new_caption.append("The caption of the {} video clip is:{} \n".format(order[index], caption))
        
        qs = " ".join(new_caption)

        # qs = "Write a summary of the following, including as many key details as possible." + qs
        qs = "You need to write a summary of the following, including as many key details as possible into one sentence." + qs
        conv = conv_templates["qwen_1_5_summarize"].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        summarize_prompt = conv.get_prompt()
        print(summarize_prompt)
        # print(question)
        # captioning_input_ids = tokenizer_image_token(captioning_prompt, summarizer_tokenzier, IMAGE_TOKEN_INDEX, return_tensors='pt')
        summarize_ids = torch.tensor(tokenizer(summarize_prompt).input_ids , dtype=torch.long)
        summarize_ids = summarize_ids.unsqueeze(0).cuda()
        return summarize_ids
    
    summarize_ids = make_summary_prompt(caption_list)
    time_1 = time.time()
    with torch.no_grad():
        output_ids = model.generate_with_image_embedding(
            summarize_ids,
            image_embeddings= None,
            modalities=["video"],
            # question_ids=ques_ids,
            # modalities="image",
            do_sample=True ,
            temperature=0.1,
            top_p=None,
            num_beams=1,
            max_new_tokens=128,
            use_cache=False)

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    time_2 = time.time()
    print("Out:{}".format(outputs))
    print("time spend:{}".format((time_2 - time_1)))
    

if __name__ == "__main__":
    # test_phi()
    test_longva_only_text()