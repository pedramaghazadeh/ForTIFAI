import os
import re
import json

from openai import OpenAI
from tqdm import tqdm
from datasets import load_dataset, load_from_disk

client = OpenAI()

def prepare_data(path='/Users/pedramaghazadeh/Desktop/data/babiqa_small'):
    if (path is not None) and (not os.path.isdir(path)):
        print("Downloading and processing dataset...")
        dataset = load_dataset('facebook/babi_qa', 'en-qa1')
        dataset.save_to_disk(path)
    else:
        print("Dataset already downloaded and processed")
        dataset = load_from_disk(path)
    return dataset

def process_wikitext(dataset):
    # Extracting paragraphs from WikiText
    paragraphs = []
    pattern = r" = =."
    begin_line = 1
    cnt = 0

    while(begin_line < len(dataset["train"])):
        header = dataset["train"][begin_line]["text"]
        body = header
        end_line = begin_line + 1

        while(end_line < len(dataset["train"])):
            line = dataset["train"][end_line]["text"]
            if(line.startswith(" =") and not re.match(pattern, line)):
                break
            body += line
            end_line += 1

        if end_line - begin_line <= 10:
            begin_line = end_line
            continue
        paragraphs.append((body, begin_line))
        begin_line = end_line
        cnt += 1

    print(len(paragraphs))
    return paragraphs

def process_tinystories(dataset):
    return dataset["validation"]["text"]

def process_babi(dataset):
    processed_data = []
    for data in dataset:
        print(data)
        story = data['story']
        context = ""
        for ind in range(len(story["text"])):
            text = story["text"][ind]
            if text[-1] == '?':
                # Question found
                processed_data.append({"context": context + text, "question": text, "answer_correct": story["answer"][ind]})
            else:
                context += text + " "
    return processed_data

questions = []

dataset = prepare_data()["train"]
print(len(dataset))
print(dataset[0])
data = process_babi(dataset)

for topic in tqdm(data):
    try:
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": """I want you to help me create a dataset. Given multiple sentences, there is a simple question at the end provided with a correct answer. I want you to create a simple WRONG answer that is different from the correct answer and is not in anyways synonymous to the correct answer.
                {"data":
                [{
                        "context": "Mary moved to the bathroom. John went to the hallway. Where is Mary?",
                        "answer_correct": "bathroom",
                        "answer_wrong": "hallway",
                    },
                ]}
                Put all the entries in a json list (similar to a list of python dictionaries). The outputs should be formatted as a json file.
                """
                },
                {
                    "role": "user",
                    "content": f"Here is the text:\ncontext: {topic["context"]},\n answer_true: {topic["answer_correct"]}"
                }
            ],
            response_format={"type": "json_object"},
        )
        # completion = client.chat.completions.create(
        #     model="gpt-4o-mini",
        #     messages=[
        #         {"role": "system", "content": """I want you to help me create a dataset. Here the task is given a paragraph create a two choice question. 
        #         The format of the question is: a context, one following statement for the context which it is false according to the body of the text and one following statement that is true according to the body of the text. 
        #         The goal is for an LLM to decide between "context + sentence_true" and "context + sentence_false". 
        #         The context needs to be short and the following sentences need to be almost identical except for key statements and factual keywords. Any model that was trained on the body of the text shoud be able to figure out the correct sentence given a short context, even though both sentences are good continutations. Strongly avoid the sentences that are only different in numerical values or dates as they are not good examples for our purpose. Do not give away the answer in the context. Focus on high quality contexts and sentences.
        #         Here is an example:
        #         The Gambia women 's national football team represents the Gambia in international football competition . The team , however , has not competed in a match recognised by FIFA , the sport 's international governing body , despite that organised women 's football has been played in the country since 1998 . The Gambia has two youth teams , an under-17 side that has competed in FIFA U-17 Women 's World Cup qualifiers , and an under-19 side that withdrew from regional qualifiers for an under-19 World Cup . The development of a national team faces challenges similar to those across Africa , although the national football association has four staff members focusing on women 's football .
                
        #         = = The team = =
        #         In 1985 , few countries had women 's national football teams . While the sport gained popularity worldwide in later decades , the Gambia 's national team only played its first game in 2007 . That game was not FIFA-recognised . As of March 2012 , the team was unranked by FIFA , and as of the following month the Gambia had not played in a FIFA-sanctioned match . The team has not participated in major regional and international tournaments , including the Women 's World Cup , the 2010 African Women 's Championship or the 2011 All-Africa Games .
        #         The country did not have a FIFA-recognised youth national team until 2012 , when the Gambia under-17 women 's team competed in Confederation of African Football qualifiers for the FIFA U-17 World Cup , to be held in Azerbaijan in September 2012 . The Gambia had fielded an under-17 team of 24 players , narrowed from an initial pool of 49 young women . Two girls from the SOS Children ’ s Village Bakoteh were chosen as a members of the team . The Gambia first played Sierra Leone in a pair of qualifying matches for the tournament . Gambia won the first match 3-0 in Banjul , the Gambia 's capital . The return match was delayed in for 24 hours and played in Makeni . The Gambia beat Sierra Leone 4-3 to qualify for the final round . The Gambia then beat Tunisia 1-0 at home and won 2-1 in Tunisia . Adama Tamba and Awa Demba scored the Gambia 's goals . Tunisia 's only goal was a Gambian own goal . The win qualified Gambia for the 2012 Azerbaijan World Cup .
        #         The Gambia also has an under-19 team that was to play in the African Women 's U-19 Championship in 2002 . The Gambia 's first match was against Morocco , but the team withdrew from the competition .
                
        #         = = Background and development = =
        #         The development of women 's football in Africa faces several challenges , including limited access to education , poverty amongst women , inequalities and human rights abuses targeting women . Funding is another issue impacting the game in Africa , where most financial assistance comes from FIFA and not national football associations . Another challenge is the retention of football players . Many women footballers leave the continent to seek greater opportunity in Europe or the United States .
        #         Gambia 's national football association was founded in 1952 , and became affiliated with FIFA in 1968 . Football is the most popular women 's sport in the country , and was first played in an organized system in 1998 . A national competition was launched in 2007 , the same year FIFA started an education course on football for women . Competition was active on both the national and scholastic levels by 2009 . There are four staffers dedicated to women 's football in the Gambia Football Association , and representation of women on the board is required by the association 's charter .
                
        #         Output:
        #         {"data":
        #         [{
        #                 "context": "The Gambia had an under-17 women's team that competed in Confederation of African Football qualifiers for the FIFA U-17 World Cup.",
        #                 "sentence_true": "The team qualified for the 2012 World Cup in Azerbaijan after defeating Sierra Leone and Tunisia.",
        #                 "sentence_false": "The team failed to qualify for the 2012 World Cup in Azerbaijan after losing to Sierra Leone and Tunisia.",
        #                 "location": 214
        #             },
        #         ]}
        #         Put the location field as 0 for all entries.
        #         Given some text consisting of multiple paragraphs, please provide 2 such entries with context, sentence true and false. Make sure the true and false statements only differ on key points and key words not in sentence structure.
        #         Put all the entries in a json list (similar to a list of python dictionaries). The outputs should be formatted as a json file.
        #         """
        #         },
        #         {
        #             "role": "user",
        #             "content": f"Here is the text:\n{topic[0]}"
        #         }
        #     ],
        #     response_format={"type": "json_object"}
    except Exception as e:
        print(f"GPT-API failed on line {topic[1]}")
        print(f"Error faced is {e}")
        continue
    try:
        subject_json = json.loads(completion.choices[0].message.content)
        for question in subject_json["data"]:
            questions.append(question)
            print(topic, "\n", question)
    except Exception as e:
        print(f"GPT-API didn't return a valid JSON file on line {topic[1]}")
        print(f"Error faced is {e}")
        continue

with open('/Users/pedramaghazadeh/Desktop/babi_dataset_gpt_mini.json', 'w') as f:
    json.dump(questions, f, indent=4)