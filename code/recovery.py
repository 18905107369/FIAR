from transformers import RobertaTokenizer, T5ForConditionalGeneration
import json
import re
from tqdm import tqdm
import time
from sklearn.cluster import KMeans
import numpy as np
import json
import csv
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import pandas as pd
from gensim.models import CoherenceModel
from gensim.corpora import Dictionary
from contextualized_topic_models.models.ctm import CombinedTM
from contextualized_topic_models.utils.data_preparation import TopicModelDataPreparation
from contextualized_topic_models.utils.preprocessing import WhiteSpacePreprocessing
from nltk.corpus import stopwords as stop_words
from transformers import RobertaTokenizer, T5EncoderModel
import time
class DataLoader:
    def __init__(self) -> None:
        self.datas = {}


    # def load_from_file(self, file_path, data_type):
    #     """
    #     """
    #     data_dict = {}
    #     with open(file_path, "r", encoding="UTF-8") as f:
    #         raw_data = dict(json.load(f))
    #     for comp, comp_info in raw_data.items():
    #         for class_name, class_info in comp_info.items():
            
    #             class_methods = {}
    #             for method_name, method_body in class_info["method"].items():
    #                 method_code = method_name + "()" + method_body
    #                 class_methods[method_name] = method_code
    #             data_dict[class_name] = { "methods":class_methods}
                
    #     self.datas[data_type] = data_dict


    def load_from_file(self, file_path, data_type):
        """
        """
        data_dict = {}
        with open(file_path, "r", encoding="UTF-8") as f:
            raw_data = dict(json.load(f))
        for comp, comp_info in raw_data.items():
            for class_name, class_info in comp_info.items():
                # print(class_name)
                class_body = class_info["body"]
                class_methods = {}
                for method_name, method_body in class_info["method"].items():
                    method_code = method_name + "()" + method_body
                    class_methods[method_name] = method_code
                    # class_body.remove(method_code)
                data_dict[class_name] = {"body": class_body, "methods":class_methods}
        self.datas[data_type] = data_dict

class Code2TextModel:
    def __init__(self, data_loader: DataLoader, t5model: T5ForConditionalGeneration, t5tokenizer: RobertaTokenizer, max_length = 20) -> None:
        self.model = t5model
        self.tokenizer = t5tokenizer
        self.max_length = max_length
        self.data_loader = data_loader
        self.processed_data = None

    def process(self, code):
        if len(code) > 512:
            code = code[:512]
        input_ids = self.tokenizer(code, return_tensors="pt").input_ids
        generated_ids = self.model.generate(input_ids, max_length=self.max_length)
        return self.tokenizer.decode(generated_ids[0], skip_special_tokens=True).replace("\n", " ")

    # def process_all(self):
    #     self.processed_data = {}
    #     for data_type, data_info in self.data_loader.datas.items():
    #         print(f"Processing set {data_type} of all data...")
    #         current_type = {}
    #         for class_name, class_info in tqdm(data_info.items()):
    #             methods_processed = {k:self.process(tm) for k,tm in class_info["methods"].items()}
    #             current_type[class_name] = {"methods":methods_processed}
    #             # print(f"\nClass {class_name} finished!")
    #             # print(f"    Body Description: {body_processed}")
    #             # for k,x in methods_processed.items():·
    #             #     print(f"    Method {k}() Description: {x}")
    #         self.processed_data[f"Data: {data_type}"] = current_type

    def process_all(self):
        self.processed_data = {}
        for data_type, data_info in self.data_loader.datas.items():
            print(f"Processing set {data_type} of all data...")
            current_type = {}
            for class_name, class_info in tqdm(data_info.items()):
                body_processed = self.process(class_info["body"])
                methods_processed = {k:self.process(tm) for k,tm in class_info["methods"].items()}
                current_type[class_name] = {"body": body_processed, "methods":methods_processed}
                # print(f"\nClass {class_name} finished!")
                # print(f"    Body Description: {body_processed}")
                # for k,x in methods_processed.items():
                #     print(f"    Method {k}() Description: {x}")
            self.processed_data[f"Data: {data_type}"] = current_type

# In[2]:


# read data without preprocessing since we are going to use WhiteSpacePreprocessingStopwords()
def load_data_without_preprocess(result):
    raw_class_names = []
    raw_data_set = []
    # with open(pathname, "r", encoding="UTF-8") as f:
    #     raw_data = dict(json.load(f))
    raw_data = dict(json.loads(result))
    for data_set, data_info in raw_data.items():
        for class_name, class_info in data_info.items():
            raw_class_names.append(class_name)
            raw_data_set.append(class_info["body"] + " " + " ".join(class_info["methods"]))
    return raw_data_set, raw_class_names

# process a list of documents with sp
def pre_preocessData(documents, process_class = WhiteSpacePreprocessing, sw = None):
    sp = process_class(documents)
    preprocessed_documents, unpreprocessed_corpus, vocab = sp.preprocess()
    return preprocessed_documents, unpreprocessed_corpus, vocab

# use a given sentence transformer model to fit the data and then fit the combined tm
# finally return the topic lists with first 12 words
def combined_topic_modeling(documents, 
                            topic_nums = 5, 
                            topic_preparing = TopicModelDataPreparation("all-mpnet-base-v2")
):
    sw = list(stop_words.words("english"))
    preprocessed_documents, unpreprocessed_corpus, vocab = pre_preocessData(documents,sw = sw)
    traning_dataset = topic_preparing.fit(text_for_contextual=unpreprocessed_corpus, text_for_bow=preprocessed_documents)
    ctm = CombinedTM(bow_size=len(topic_preparing.vocab), contextual_size=768, n_components=topic_nums, num_epochs=10)
    ctm.fit(traning_dataset)
    topic_list = ctm.get_topic_lists(12)
    for i, topic in enumerate(topic_list):
        # print(f"Topic {i+1}: {topic}")
        print()
    return ctm, topic_preparing, traning_dataset
        


# In[3]:
function_info={}
if __name__=='__main__':

    data_loader = DataLoader()
    #  input
    data_loader.load_from_file("info.json", 1)

    start=time.time()

    t5model = T5ForConditionalGeneration.from_pretrained('Salesforce/codet5-base-multi-sum')
    t5tokenizer = RobertaTokenizer.from_pretrained('Salesforce/codet5-base-multi-sum')

    model = Code2TextModel(data_loader, t5model, t5tokenizer)

    model.process_all()

    end=time.time()

    print(end-start)

    # with open("result.json", 'w', encoding="UTF-8") as f:
    #     f.write(json.dumps(model.processed_data))

    result1=json.dumps(model.processed_data)

    all_preds = {}
    all_cont_vectors = {}
    all_bow_vectors = {}

    print(f"\n == RUN for topic num: {2}")
    raw_data_code2, raw_name_code2 = load_data_without_preprocess(result1)
    ctm_code2, tp_code2, td_code2 = combined_topic_modeling(raw_data_code2, topic_nums=2)
    # topic_predictions_code2 = ctm_code2.get_thetas(td_code2, n_samples=10)
    # topic_number_code2 = np.argmax(topic_predictions_code2, axis=1)
    result_topic = {}
    for i, class_name in enumerate(raw_name_code2):
        all_cont_vectors[class_name] = td_code2.X_contextual[i, :]

    for class_name in all_cont_vectors:
        all_cont_vectors[class_name] = [float(x) for x in all_cont_vectors[class_name]]

    # with open(f"function_info.json", "w", encoding="UTF-8") as f:
    #     f.write(json.dumps(all_cont_vectors))
   
    function_info=all_cont_vectors


   

    start=time.time()

    tokenizer = RobertaTokenizer.from_pretrained('Salesforce/codet5-base')
    model = T5EncoderModel.from_pretrained('Salesforce/codet5-base')

    import json
    data_path = "info.json"
    with open(data_path, "r") as f:
        data_dict = json.loads(f.read())

    model.eval()
    result_Dict = {}
    code_dict = {}
    path_dict = {}


    for comp, comp_info in data_dict.items():
        for class_name, class_info in comp_info.items():
            imports = class_info["deps"]
            # for dep in imports:
            #     deps=dep.split(" ")
            #     dep=deps[0].rsplit(".",1)[-1].rsplit("/",1)[-1]+" "+ deps[1] +" "+deps[-1].rsplit(".",1)[-1].rsplit("/",1)[-1]
            import_text = " ".join(imports)
            input_ids = tokenizer(import_text, return_tensors="pt").input_ids   
            out = model(input_ids = input_ids)
            vector_list = out.last_hidden_state[0,-1,:].tolist()
            result_Dict[class_name] = vector_list

            # body_text = class_info["body"][:512]
            # input_ids2 = tokenizer(body_text, return_tensors="pt").input_ids   
            # out2 = model(input_ids = input_ids2)
            # vector_list2 = out2.last_hidden_state[0,-1,:].tolist()
            # code_dict[class_name] = vector_list2

            # print(class_info["className"])
            path_text = " ".join(class_info["directory"])
            input_id3 = tokenizer(path_text, return_tensors="pt").input_ids   
            out3 = model(input_ids = input_id3)
            vector_list3 = out3.last_hidden_state[0,-1,:].tolist()
            path_dict[class_name] = vector_list3
        
    # with open("dependency_info.json", "w") as f:
    #     json.dump(result_Dict, f)

    dependency_info=result_Dict
    # for comp, comp_info in result1.items():
    #     for class_name, class_info in comp_info.items():

    #         body_text = class_info["body"] + " " + " ".join(class_info["methods"])
    #         input_ids2 = tokenizer(body_text, return_tensors="pt").input_ids   
    #         out2 = model(input_ids = input_ids2)
    #         vector_list2 = out2.last_hidden_state[0,-1,:].tolist()
    #         code_dict[class_name] = vector_list2
    # function_info=code_dict
    # with open("bash_code_info3.json", "w") as f:
    #     json.dump(code_dict, f)

    # with open("directory_info.json", "w") as f:
    #     json.dump(path_dict, f)
    directory_info=path_dict
    # this prints "user: {user.name}"
    end=time.time()

    print(str(end-start))

    #!/usr/bin/env python
    # coding: utf-8


    import_info = result_Dict


    code_info = function_info


    path_info = path_dict


    class_list = []
    vec_list = []


    for class_name in import_info.keys():
        import_vec = import_info[class_name]
        code_vec = code_info[class_name]
        path_vec = path_info[class_name]
        final_vec = import_vec + code_vec + path_vec
        vec_list.append(final_vec)
        class_list.append(class_name)

    import numpy as np
    from sklearn.decomposition import PCA
    if len(vec_list) >32:
        pca = PCA(n_components=32)
        vec_pca = pca.fit_transform(vec_list)

    k=4

    kmeans = KMeans(n_clusters=k, random_state=0, n_init="auto").fit(vec_list)
    result = dict(zip(class_list, [int(x) for x in kmeans.labels_]))
    inv_result = {}
    for cn, cp in result.items():
        if cp not in inv_result:
            inv_result[cp] = [cn]
        else:
            inv_result[cp].append(cn)
        # with open(f"final_kmeans_comp2class_{k}.json", "w") as f:
        #     json.dump(inv_result, f)
        with open(f"recovery_result.json", "w") as f:
            json.dump(result, f)
