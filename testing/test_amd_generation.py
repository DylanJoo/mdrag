import torch
from datasets import load_dataset 
import time
from tqdm import tqdm

if __name__ == '__main__':
    dataset = load_dataset('BeIR/scidocs', 'corpus', keep_in_memory=True)['corpus']
    documents = dataset['text'][:64]
    guideline = "- 5: The context is highly relevant, complete, and accurate.\n- 4: The context is mostly relevant and complete but may have minor gaps or inaccuracies.\n- 3: The context is partially relevant and complete, with noticeable gaps or inaccuracies.\n- 2: The context has limited relevance and completeness, with significant gaps or inaccuracies.\n- 1: The context is minimally relevant or complete, with substantial shortcomings.\n- 0: The context is not relevant or complete at all."
    template_rating = "Instruction: {INST}\n\nGuideline:\n{G}\n\nQuestion: {Q}\n\nContext: {C}\n\n{PREFIX}" 
    instruction_rating = "Determine whether the question can be answered based on the provided context? Rate the context with on a scale from 0 to 5 according to the guideline below. Do not write anything except the rating."
    def prompt_rating_gen(INST="", Q="", C="", PREFIX="Rating:"):
        p = template_rating
        p = p.replace("{G}", guideline)
        p = p.replace("{INST}", INST).strip()
        p = p.replace("{Q}", Q)
        p = p.replace("{C}", C)
        p = p.replace("{PREFIX}", PREFIX).strip()
        return p

    prompts = list(
            prompt_rating_gen(
                INST=instruction_rating, 
                Q=' What is the central role that software development plays in the delivery',
                C=d,
            ) for d in documents
    )

    # model_name_or_path = "meta-llama/Meta-Llama-3-8B-Instruct"
    model_name_or_path = "meta-llama/Meta-Llama-3.1-70B-Instruct"

    from vllm import LLM
    from vllm import SamplingParams
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.95,
        skip_special_tokens=False,
        max_tokens=640,
    )
    model = LLM(
        model_name_or_path, 
        dtype='half',
        enforce_eager=True,
        tensor_parallel_size=2,
        pipeline_parallel_size=4
    )

    print("Start")
    start_ = time.time()
    for i, prompt in enumerate(prompts):
        prompt = """Instruction: Break down the given document into 2-3 standalone passages of approximately 200 words each, providing essential context and information. Use similar wording and phrasing as the original document. Write each passages within `<p>` and `</p>` tags.\n\nDocument:  You might have heard already that Tina Fey and Amy Poehler "went there" at the Golden Globes on Sunday -- that they pulled no punches when it came to the sexual assault accusations against Bill Cosby, that they poked fun at the comedian's alleged history of drugging and raping nearly two dozen women. It was a good and uncomfortable move. As my colleague Sonia Saraiya noted, the jokes were "bold, brash, and refreshingly on-the-nose," arguably the only way of handling such a sensitive topic in the first ten minutes of a Hollywood awards show. Ultimately, though, Fey and Poehler didn't have to bring up the Cosby allegations at all; judging from the audience's uncomfortable response, people might have been more comfortable if they hadn't. And that's exactly why the jokes were necessary, even if they weren't Fey or Poehler's best. Mimicking Cosby's voice isn't exactly the most original humor, but the way the hosts turned the embattled star into a punchline -- instead of the women who have accused him of assault -- marks a fresh approach to the grotesque un-funniness of violence against women. It's similar to the approach comedian Hannibal Buress took when he brought the decades-old Cosby rape allegations back into the public eye at the end of last year -- by making a rape joke that doesn't joke about rape.\nPassages:\n<p>""".strip()
        if i == 0:
            print(prompt)

        start = time.time() 
        outputs = model.generate(prompt, sampling_params)[0].outputs[0].text
        end = time.time() 
        print(outputs)
        print(f"document {i}: {(end - start)} | Elapsed: {end - start_} s")

    print('end')
