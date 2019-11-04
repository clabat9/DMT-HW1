# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 14:43:14 2019

@author: claba
"""

from whoosh.index import create_in
from whoosh.fields import Schema,ID,TEXT
from whoosh.analysis import StemmingAnalyzer, StandardAnalyzer
from whoosh.qparser import MultifieldParser,SimpleParser
from whoosh import scoring
import re
import pandas as pd
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

###
### Plotting function
###
def dictionary_plot(a_dictionary, save_dir = r"C:\Users\claba\Desktop\DMT works\HW_1"):
    jet= plt.get_cmap('nipy_spectral')
    colors = iter(jet(np.linspace(0, 1,len(a_dictionary))))
    figure(num=None, figsize=(40, 32), dpi=100, facecolor='w', edgecolor='k')
    for y in a_dictionary:
        x = range(1, len(a_dictionary[y])+1)
        plt.plot(x, a_dictionary[y], marker = "o",markersize = 3, color=next(colors), label = y) 
    plt.legend(loc='lower right', prop={'size': 25})
    plt.savefig(save_dir+'\\nDCg.png')
    plt.show()
    

###
###Function to parse document (basic use of RegExs)
###
def doc_retriver(doc_path):
    with open(doc_path) as f:
        doc = f.read()
        title = re.findall(r"(?<=<title>)[\w\W]*(?=</title>)", doc)[0].replace("\n"," ")
        content = re.findall(r"(?<=<body>)[\w\W]*(?=</body>)", doc)[0].replace("\n"," ")
    return title,content


###
### Metrics implementations, not too much to comment, just coding the formulas.
###

def mrr(one_se_results,ground_truth):
    Q = len(ground_truth)
    numerator = []
    for query in ground_truth:
        for idx,el in enumerate(one_se_results[:,query]):
            if el in ground_truth[query]:
                numerator.append(1/(idx+1))
                break
    return(sum(numerator)/Q)
    
    
    
def r_precision(one_se_results, ground_truth):
    query2rprecision = {}
    for query in ground_truth:
        len_gt = len(ground_truth[query])
        counter = 0
        for el in one_se_results[:, query][:len_gt]: 
            if el in ground_truth[query]:
                counter += 1
        query2rprecision[query] = counter/len_gt
    return query2rprecision


def nDCG(one_se_results, ground_truth,k = 50):
    query2nDCG = {}
    for query in ground_truth:
      tmp_num = []
      len_gt = len(ground_truth[query])
      for idx,el in enumerate(one_se_results[:, query]):
          if idx > k:
              break
          if el in ground_truth[query]:
                tmp_num.append(1/np.log2((idx+1)+1))
      dcg = sum(tmp_num)
      idcg= sum([1/np.log2((idx+1)+1) for idx in range(min(k,len_gt)+1)])
      query2nDCG[query] = dcg/idcg
    return query2nDCG


################################################################################
###            EXECUTION AND EVALUATIONS OF SEARCH ENGINE                    ###  
###  The parameters are pretty intuitive, just a small remark about the fact ###
###  that choosing a multifield search obviously makes unuseful the          ###
###  "only_title_flag"  that allows to select only title if ==1 or only      ###
###  content otherwise.                                                      ###
################################################################################

def search_engine( analyzer = StemmingAnalyzer(), max_res = 150, multifield_flag = 1, \
                  only_title_flag = 0, \
                  directory_containing_the_index  = r"C:\Users\claba\Desktop\DMT works\HW_1\Index_part_1", \
                  query_dir = r"C:\Users\claba\Desktop\DMT works\HW_1\part_1\Cranfield_DATASET\cran_Queries.tsv", \
                  gt_dir = r"C:\Users\claba\Desktop\DMT works\HW_1\part_1\Cranfield_DATASET\cran_Ground_Truth.tsv", \
                  doc_dir = r"C:\Users\claba\Desktop\DMT works\HW_1\part_1\Cranfield_DATASET\DOCUMENTS\\", \
                  conf_label = "Not Specified",
                  mrr_eps = .32, \
                  k_interval_for_nDCG = range(1,151)):
   
    
    ###
    ### Create a Schema 
    ###
    schema = Schema(id=ID(stored=True), \
                    title = TEXT(stored=False, analyzer=analyzer),content=TEXT(stored=False, analyzer=analyzer))
    
    ###
    ### Create an empty-Index 
    ### according to the just defined Schema ;)
    ### 
    ix = create_in(directory_containing_the_index, schema)
    
    
    ###
    ### Get the query set (reset index due to missing values in the IDs)
    ###
    query_set = pd.read_csv(query_dir, engine = "python", sep = "\t", index_col="Query_ID").reset_index()
    
    
    ###
    ### Get the ground truth (little manipulation to group by query and allign IDs)
    ###
    gt_tmp = pd.read_csv(gt_dir, engine = "python", sep = "\t")
    gt_tmp = gt_tmp.groupby('Query_id')['Relevant_Doc_id'].apply(lambda x: x.tolist()).to_dict()
    gt = defaultdict(list)
    j = 1
    for i in range(len(gt_tmp)):
        while(gt[i] == []):
            try:
                gt[i] = gt_tmp[j]
                j+=1
            except KeyError:
                j += 1
    
    
    
    number_of_queries = len(query_set)
    num_of_docs = 1400
    
    ###
    ### We'll iterate on the following lists to swicth SE scoring function and get their names
    ###
    scoring_functions_list = [scoring.PL2(), scoring.Frequency(), scoring.BM25F(), scoring.TF_IDF()]
    scoring_name = [re.findall(r"(?<=scoring\.)[\w\W]*(?=object)", str(score))[0] for score in scoring_functions_list]
    
    
    ###
    ### Fill the Index
    ###
    writer = ix.writer()
    for doc in range(num_of_docs):
        id_ = str(doc+1)
        title,content = doc_retriver(doc_dir+"______"+str(doc+1)+".html")
        writer.add_document(id=id_, title = title, content = content)
    writer.commit()
    
    
    
    ###
    ### This """tensor""" allows to store all the results we need. It's dimension are #ResultsX#QueriesX#SE_config
    ###
    results_mat = np.zeros([max_res,number_of_queries,len(scoring_functions_list)])
    
   
    evaluations_summary = {} # Dict to store MRR and R-Precision Distro sumamries
    ndcg = defaultdict(list) # Def Dict that will contain nDCG values for varying K values for all MRR >.32 SEs

    ###
    ### Run the SEs
    ###
    for idx_s,scorer in enumerate(scoring_functions_list):
        for idx,query in enumerate(query_set["Query"]):
            
            input_query = query
            
            ###
            ### Select a Scoring-Function
            ###
            scoring_function = scorer
            
            ###
            ### Create a QueryParser for 
            ### parsing the input_query based on user SE choosen configuration.
            ###
            if multifield_flag:
                qp = MultifieldParser(["title","content"], ix.schema)
                parsed_query = qp.parse(input_query)# parsing the query
            else:
                if only_title_flag:
                    qp = SimpleParser("title", ix.schema)
                    parsed_query = qp.parse(input_query)# parsing the query
                else:
                    qp = SimpleParser("content", ix.schema)
                    parsed_query = qp.parse(input_query)# parsing the query
                
            ###
            ### Create a Searcher for the Index
            ### with the selected Scoring-Function 
            ###
            searcher = ix.searcher(weighting=scoring_function)
            
            ###
            ### Perform a Search and store results
            ###
            results = searcher.search(parsed_query, limit=max_res)
            results_mat[0:len(results),idx,idx_s] = [hit["id"] for hit in results]
            searcher.close()
        mrr_res = mrr(results_mat[:,:,idx_s],gt)
        
        if mrr_res >= mrr_eps:
            
            ###
            ### Compute and summarize R-precision distro
            ###
            r_res = r_precision(results_mat[:,:,idx_s],gt)
            mean = np.mean(list(r_res.values()))
            first_q = np.percentile(list(r_res.values()),25)
            third_q = np.percentile(list(r_res.values()),75)
            median = np.median(list(r_res.values()))
            minr = min(list(r_res.values()))
            maxr = max(list(r_res.values()))
            evaluations_summary[conf_label+","+scoring_name[idx_s]] = [mrr_res,mean,minr,first_q,median,third_q,maxr]
            
            ###
            ### Compute nDCG@k for varying k and for each scoring function
            ###
            for k in k_interval_for_nDCG:
                tmp_res = np.mean(list(nDCG(results_mat[:,:,idx_s],gt,k = k).values()))
                ndcg[conf_label+","+scoring_name[idx_s]].append(tmp_res)
            
        else:
            evaluations_summary[conf_label+","+scoring_name[idx_s]] = [mrr_res]
        
        ###
        ### Just to see what's happening
        ###
        print("Configuration:"+conf_label+","+scoring_name[idx_s]+"==> MRR = "+str(mrr_res))
        
    return evaluations_summary, ndcg # The evaluation result, obviously, contains oly MRR for <.32 SEs 
                                     # and also R-precision data for others (in the requested order)
  

   
###
### Now that we have all the functions we need, we can test SE's performances by 
### playing a little bit with the parameters of the "search_engine" functions.
### Note that each configurations is tested on 4 different scoring functions.
###
    
res = []  # To store the results (list of dicts)
ndcg_tot = [] # To store nDCG values

###        
### Search over only "title" field with "StemmingAnalyzer":
###       
mrr_r,ndcg = search_engine( multifield_flag = 0, only_title_flag = 1, conf_label = "Title,Stemming") 
res.append(mrr_r)
ndcg_tot.append(ndcg)

###
### Search over only "title" field with "StandardAnalyzer":
###        
mrr_r,ndcg = search_engine( multifield_flag = 0, only_title_flag = 1, analyzer=StandardAnalyzer(), conf_label = "Title,Standard")
res.append(mrr_r)
ndcg_tot.append(ndcg)

###
### Search over only "content" field with "StemmingAnalyzer":
###        
mrr_r,ndcg = search_engine( multifield_flag = 0, only_title_flag = 0, conf_label = "Content,Stemming")
res.append(mrr_r)
ndcg_tot.append(ndcg)
    
###
### Search over only "content" field with "StandardAnalyzer":
###        
mrr_r,ndcg = search_engine( multifield_flag = 0, only_title_flag = 0,analyzer=StandardAnalyzer(), conf_label = "Content,Standard")
res.append(mrr_r)
ndcg_tot.append(ndcg)

###        
### Search over both "title" and "content" fields with "StemmingAnalyzer":
###        
mrr_r,ndcg = search_engine( multifield_flag = 1, conf_label = "Title&Content,Stemming")
res.append(mrr_r)
ndcg_tot.append(ndcg)

###        
### Search over both "title" and "content" fields with "StandardAnalyzer":
###        
mrr_r,ndcg = search_engine( multifield_flag = 1,analyzer=StandardAnalyzer(), conf_label = "Title&Content,Standard")
res.append(mrr_r)
ndcg_tot.append(ndcg)




###
### The following lines are useful to build requested tables and plots
###

###
### Some Manipulations in order to better organize and visualize the results by merging all the obtained results
###

mrr_dic = {}
for r in res: mrr_dic.update(r)

ndcg_dic = {}
for r in ndcg_tot: ndcg_dic.update(r)    



all_mrr = {} # Dict that will be used to build a pd series in order to better visualize the results (and export them)
complete_frame = pd.DataFrame(columns = ["MRR","R-Precision Mean","R-Precision Min", \
                                    "R-Precision first Quartile", "R-Precision Median", \
                                    "R-Precision third quartile", "R-Precision Max"] ) #DataFrame summarizing 
                                                                                       # R-precision distro

###
### Building the requested tables
###                                                                                       
                                                                                     
for el in mrr_dic:
    all_mrr[el] = mrr_dic[el][0]
    if len(mrr_dic[el])>1: #If MRR > .32...
        complete_frame.loc[el] = mrr_dic[el]
complete_frame.index.name = "Configuration [Field,Analyzer,Scoring]"
    

mrr_series  = pd.Series(all_mrr, name = "MRR")
mrr_series.index.name = "Configuration [Field,Analyzer,Scoring]"
mrr_series.reset_index()

mrr_series.to_csv("MRR.csv")
complete_frame.to_csv("distro.csv",)  
dictionary_plot(ndcg_dic) # Visualize and save nDCG plots


    


    
    
    
    
     
    
    
