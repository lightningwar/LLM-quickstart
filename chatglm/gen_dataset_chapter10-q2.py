import csv
import random
from zhipuai import ZhipuAI

client = ZhipuAI(api_key='xxx')
glm_model = 'glm-4-0520'


def gen_data(raw_content):
    """
    使用LangChain GPT-3.5调用处理单个数据样例。

    :param raw_content: 原始数据样例。
    :return: GPT-3.5模型生成的内容。
    """
    # 系统消息定义背景和任务
    system_content = """你是中国古典哲学大师，尤其擅长周易的哲学解读。

接下来，你收到的都是关于周易卦象的解释，你需要整理润色，并生成用于大模型训练的内容和格式，但是不能出现大模型、润色、以下、总结等字眼。

示例输入：

师卦，此卦是异卦相叠，下卦为坎，上卦为坤。“师”指军队。坎为水、为险；坤为地、为顺，喻寓兵于农。兵凶战危，用兵乃圣人不得已而为之，但它可以顺利无阻碍地解决矛盾，因为顺乎形势，师出有名，故能化凶为吉。占得此卦，对于军事上率师出征非常有利，必无灾祸。师卦是天马出群之卦，以寡伏众之象。
师卦位于讼卦之后，《序卦》之中这样解释道：“讼必有众起，故受之以师。师者，众也。”争讼的人越来越多，以致形成了军队。

期待结果：

content:"师卦"
summary:"在周易中，师卦是一个极具深意的卦象，它由两个异卦相叠组成：下卦坎（水）和上卦坤（地）。这一卦象代表“师”，即军队，寓意着兵力和农力的结合。在这里，坎卦象征着水和险难，而坤卦象征着地和顺从，暗示着通过将军事力量安置于民间，可以在必要时顺利调动。

师卦的核心哲学是：虽然兵力代表着危险和战争，但其使用应当是圣人不得已而为之的最后手段。在正确的情况下，军事力量可以顺应形势，将危险转化为吉祥。因此，在军事策略上，此卦象征着出征将会顺利，无灾祸。

师卦紧随讼卦（争讼卦），在《序卦》中解释为“讼必有众起，故受之以师”。这意味着争端激化至众多人群的参与，形成了类似军队的集体力量。"

返回格式要求：
content:"{卦名}"
summary:"{内容}"
"""
    while True:
        try:
            ai_message = client.chat.completions.create(
                model=glm_model,
                messages=[
                    {
                        'role': 'system',
                        'content': system_content
                    },
                    {
                        'role': 'user',
                        'content': raw_content
                    }
                ],
                temperature=1,
                top_p=1
            )
            break
        except:
            pass

    # 人类消息包含原始数据样例
    return ai_message.choices[0].message.content


def dataset_parser(ai_message_content):
    """
    解析由gen_data函数生成的ai_message.content，提取content和summary。

    :param ai_message_content: gen_data函数返回的文本。
    :return: 提取的content和summary。
    """
    # 分割字符串来找到content和summary的位置
    content_start = ai_message_content.find('content:"') + len('content:"')
    content_end = ai_message_content.find('"\nsummary:')
    summary_start = ai_message_content.find('summary:"') + len('summary:"')
    summary_end = ai_message_content.rfind('"')

    # 提取并存储content和summary
    content = ai_message_content[content_start:content_end].strip()
    summary = ai_message_content[summary_start:summary_end].strip()

    return content, summary


def generate_question_summary_pairs(content, summary, number, tag):
    """
    生成20对提问和总结的配对。

    :param content: 内容（例如：“蒙卦”）。
    :param summary: 内容的总结。
    :return: 包含20对提问和总结的列表。
    """
    raw_number = number
    question_summary_pairs = []

    while number:
        n = random.randint(1, 16) if number > 16 else number
        system_content = f'你是中国古典哲学大师，尤其擅长周易的哲学解读。\n\n接下来，你收到的是关于周易卦象的解释，你需要根据文本给出{n}对不同的问题和对应的答案，其中问题和答案的描述应该尽可能详细。\n\n随机数：{random.randint(0, 100000)}\n\n返回json格式如下：\n' + "[{'content':问题, 'summary':答案},{'content':问题, 'summary':答案}]"

        done = False

        while not done:
            print(content, number)

            try:
                response = client.chat.completions.create(
                    model=glm_model,
                    messages=[
                        {
                            "role": "user",
                            "content": system_content
                        },
                        {
                            "role": "user",
                            "content": summary
                        }
                    ],
                    temperature=1,
                    top_p=1
                )

                qa_pair = eval(response.choices[0].message.content.replace('json', '').replace('```', ''))
                # assert len(qa_pair) <= n

                # 创建提问和总结的配对
                question_summary_pairs += [(qa['content'], qa['summary']) for qa in qa_pair]

                done = True
            except:
                done = False

        number -= n

    question_summary_pairs = question_summary_pairs[:raw_number]
    return question_summary_pairs


# 解析 data/raw_data.txt 得到 raw_content_data 列表
raw_content_data = []
question_templates = [
            "{}代表什么？",
            "周易中的{}含义是什么？",
            "请解释一下{}。",
            "{}在周易中是什么象征？",
            "周易{}的深层含义是什么？",
            "周易的{}讲述了什么？",
            "{}是怎样的一个卦象？",
            "{}的基本意义是什么？",
            "周易中{}的解释是什么？",
            "{}在周易中代表了哪些方面？",
            "{}涉及哪些哲学思想？",
            "周易中{}的象征意义是什么？",
            "{}的主要讲述内容是什么？",
            "周易{}的核心思想是什么？",
            "在周易中，{}象征着什么？",
            "请描述{}的含义。",
            "{}在周易哲学中扮演什么角色？"
        ]
with open('data/raw_data.txt', 'r', encoding='utf-8') as file:
    content = file.read()
    data_samples = content.split('\n\n')
    for sample in data_samples:
        cleaned_sample = sample.strip()
        if cleaned_sample:
            raw_content_data.append(cleaned_sample)

    # filename_train = f'data/zhouyi_dataset_autogenerate2.csv'
    filename_train = f'data/zhouyi_dataset_train.csv'
    filename_val = f'data/zhouyi_dataset_val.csv'
    with open(filename_train, mode='w', newline='', encoding='utf-8') as f1, open(filename_val, mode='w', newline='', encoding='utf-8') as f2:
        writer1 = csv.writer(f1)
        writer2 = csv.writer(f2)
        writer1.writerow(['content', 'summary'])
        writer2.writerow(['content', 'summary'])

        # 循环遍历 raw_content_data 数据样例
        for raw_content in raw_content_data:
            # 调用 gen_data 方法得到 ai_message_content
            pairs_train = []
            for qt in question_templates:
                ai_message_content = gen_data(raw_content)
                # 解析 ai_message_content 得到 content 和 summary
                content, summary = dataset_parser(ai_message_content)

                print("Content:", content)
                print("Content:", summary)
                pairs_train.append((qt.format(content), summary))

            # 调用 generate_question_summary_pairs 得到20组 pairs
            pairs_train += generate_question_summary_pairs(content, summary, 128-len(question_templates), 'train')
            pairs_val = generate_question_summary_pairs(content, raw_content, 32, 'val')

            # 将 pairs 写入 csv 文件
            for pair in pairs_train:
                writer1.writerow(pair)

            for pair in pairs_val:
                writer2.writerow(pair)
