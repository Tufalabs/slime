from slime.utils.http_utils import post
from slime.utils.types import Sample

JUDGE_PROMPT = (
    "Check if the following <response>{response}</response> "
    "is the solution to <problem>{problem}</problem>. If its "
    "correct return 1 in <score></score> tags, otherwise "
    "return 0 in <score></score> tags."
)


async def reward(args, sample: Sample, **kwargs):
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"
    sampling_params = dict(
        temperature=args.rollout_temperature,
        top_p=args.rollout_top_p,
        top_k=args.rollout_top_k,
    )

    prompt = JUDGE_PROMPT.format(response=sample.response, problem=sample.metadata["problem"])
    payload = {
        "text": prompt,
        "sampling_params": sampling_params,
        "return_logprob": False,
    }

    output = await post(url=url, payload=payload, use_http2=args.use_http2)
    # score = extract_score(output["text"]) # ... process output from judge

    return 1.0
