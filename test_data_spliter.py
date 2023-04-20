import json
from sklearn.model_selection import train_test_split
from uuid import uuid4

def read_squad_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        squad_data = json.load(f)
    return squad_data['data']

train_data = read_squad_data("train-v2.0.json")
dev_data = read_squad_data("dev-v2.0.json")

def flatten_squad_data(squad_data):
    flattened_data = []
    for topic in squad_data:
        for paragraph in topic['paragraphs']:
            context = paragraph['context']
            for qa in paragraph['qas']:
                question = qa['question']
                if not qa['is_impossible']:
                    answer = qa['answers'][0]['text']
                    answer_start = qa['answers'][0]['answer_start']
                else:
                    answer = None
                    answer_start = -1
                flattened_data.append((context, question, answer, answer_start))
    return flattened_data

def save_data_to_json(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def save_data_to_squad_format(data, output_file_path):
    squad_format = {
        "version": "v2.0",
        "data": []
    }
    
    for context, question, answer in data:
        topic = {
            "title": "",
            "paragraphs": [
                {
                    "context": context,
                    "qas": [
                        {
                            "question": question,
                            "id": str(uuid4()),
                            "answers": [{"text": answer, "answer_start": -1}] if answer is not None else [],
                            "is_impossible": answer is None
                        }
                    ]
                }
            ]
        }
        squad_format["data"].append(topic)
    
    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(squad_format, f, ensure_ascii=False, indent=2)
        
        
        


flattened_train_data = flatten_squad_data(train_data)
train_set, test_set = train_test_split(flattened_train_data, test_size=0.1, random_state=42)

flattened_dev_data = flatten_squad_data(dev_data)
save_data_to_json(train_set, "train_set_splited.json")
save_data_to_json(test_set, "test_set_splited.json")
save_data_to_json(flattened_dev_data, "dev_set_splited.json")