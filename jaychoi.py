from openai import OpenAI

client = OpenAI()

def ask_question(question):
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant. and answer always in Korean."},
            {"role": "user", "content": question}
        ]
    )

    result = completion.choices[0].message.content
    print(result)
    return result


# ask = """
# 1. **목표:** 영화 리뷰 텍스트를 입력받아 해당 리뷰의 감정이 긍정적인지 부정적인지 판별합니다.
# 2. **출력 형식:** "긍정" 또는 "부정" 두 가지 중 하나로 출력합니다.
# 3. **5-shot 예제:** 다음 5개의 예제를 포함하여 프롬프트를 구성합니다.
#     * "이 영화는 정말 재밌고 감동적이었어요." -> "긍정"
#     * "연출이 훌륭하고 배우들의 연기도 뛰어났습니다." -> "긍정"
#     * "지루하고 재미없는 영화였습니다." -> "부정"
#     * "스토리가 엉성하고 개연성이 부족했습니다." -> "부정"
#     * "시간 낭비였습니다. 다시는 보고 싶지 않아요." -> "부정"
# 4. **테스트 입력:** 생성된 프롬프트를 사용하여 "The storyline was dull and uninspiring." 이라는 입력에 대한 결과를 예측하고 출력해주세요.
#          """

# ask_question(ask)

# ask = """
# Convert the following natural language requests into SQL queries:
# 1. "Write your Prompt": SELECT * FROM employees WHERE salary > 50000; // 연봉이 5만달러 이상인 고용인을 찾아줘
# 2. "Write your Prompt": SELECT * FROM products WHERE stock = 0; // 주식이 0인 제품을 찾아줘
# 3. "Write your Prompt": SELECT name FROM students WHERE math_score > 90; // 수학 점수가 90보다 큰 학생들을 찾아줘
# 4. "Write your Prompt": SELECT * FROM orders WHERE order_date >= DATE_SUB(CURDATE(), INTERVAL 30 DAY); // 현재 날짜보다 30일 이전부터 시작된 주문들을 검색해줘
# 5. "Write your Prompt": SELECT city, COUNT(*) FROM customers GROUP BY city; // 도시별로 고객의 수를 찾아줘

# Request: "Find the average salary of employees in the marketing department."
# SQL Query:
# """

# ask_question(ask)

# ask = """
# Solve the following problem step-by-step: 23 + 47

# 1. 십의 자리수를 각각 더합니다.
# 2. 일의 자리수를 각각 더합니다.
# 3. 1과 2의 과정에서 나온 결과를 더합니다.
# 4. 최종 결과가 맞는지 검증합니다.
# """

# result = ask_question(ask)

# # make ask for using result from previous question
# ask = f"""
# Solve the following problem step-by-step: 123 - 58

# Step-by-step solution:
# 1. 큰 수에서 작은 수를 빼기 위한 준비 작업을 합니다.
# 2. 일의 자리에서 뺄셈을 진행합니다.
# 3. 다음으로 십의 자리에서 뺄셈을 진행합니다.
# 4. 최종 결과를 합칩니다.
# """

# ask_question(ask)

# ask = """
# Step-by-step solution for add:
# 1. 십의 자리수를 각각 더합니다.
# 2. 일의 자리수를 각각 더합니다.
# 3. 1과 2의 과정에서 나온 결과를 더합니다.
# 4. 최종 결과가 맞는지 검증합니다.

# Step-by-step solution for subtract:
# 1. 큰 수에서 작은 수를 빼기 위한 준비 작업을 합니다.
# 2. 일의 자리에서 뺄셈을 진행합니다.
# 3. 다음으로 십의 자리에서 뺄셈을 진행합니다.
# 4. 최종 결과를 합칩니다.

# Solve the following problem step-by-step: 345 + 678 - 123

# Step-by-step solution:
# 1. Check response for add
# 2. Check response for subtract
# 3. Check response for result

# answer for korean, and format for terminal.
# """

# ask_question(ask)

# ask = """
# Solve the following logic puzzle step-by-step:
# Three friends, Alice, Bob, and Carol, have different favorite colors: red, blue, and green. We know that:
# 1. Alice does not like red.
# 2. Bob does not like blue.
# 3. Carol likes green.

# Determine the favorite color of each friend.

# Step-by-step solution:
# 1. Check the first clue.
# 2. Check the second clue.
# 3. Check the third clue.
# 4. Determine the favorite color of each friend.
# """

# ask_question(ask)

# ask = """
# Solve the following logic puzzle step-by-step:
# Four people (A, B, C, D) are sitting in a row. We know that:
# 1. A is not next to B.
# 2. B is next to C.
# 3. C is not next to D.

# Determine the possible seating arrangements.

# Step-by-step solution:
# 1. Check the first clue.
# 2. Check the second clue.
# 3. Check the third clue.
# 4. Determine the possible seating arrangements.
# """

# ask_question(ask)

import promptbench as pb

# print('All supported datasets: ')
# print(pb.SUPPORTED_DATASETS)

# load a dataset, sst2, for instance.
# if the dataset is not available locally, it will be downloaded automatically.
dataset_name = "gsm8k"
dataset = pb.DatasetLoader.load_dataset(dataset_name)

# print the first 3 examples
dataset[:3]

# print all supported models in promptbench
# print('All supported models: ')
# print(pb.SUPPORTED_MODELS)

# load a model, gpt-3.5-turbo, for instance.
# If model is openai/palm, need to provide openai_key/palm_key
# If model is llama, vicuna, need to provide model dir
model = pb.LLMModel(model='gpt-3.5-turbo',
                    api_key = 'openai_key',
                    max_new_tokens=150)

# print('All supported methods: ')
# print(pb.SUPPORTED_METHODS)
# print('Supported datasets for each method: ')
# print(pb.METHOD_SUPPORT_DATASET)

# load a method, emotion_prompt, for instance.
# https://github.com/microsoft/promptbench/tree/main/promptbench/prompt_engineering
method = pb.PEMethod(method='emotion_prompt',
                        dataset=dataset_name,
                        verbose=True,  # if True, print the detailed prompt and response
                        prompt_id = 1  # for emotion_prompt
                        )

results = method.test(dataset,
                      model,
                      num_samples=3 # if don't set the num_samples, method will use all examples in the dataset
                      )
results