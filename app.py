from logging import debug
from flask import Flask, app,render_template,request, url_for,redirect
import numpy as np
import pickle
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix


app = Flask(__name__)
model = pickle.load(open('model.pkl','rb'))
custitem = pickle.load(open('custitem.pkl','rb'))
custdic = pickle.load(open('custdic.pkl','rb'))
itemdic = pickle.load(open('itemdic.pkl','rb'))
mostbought = pickle.load(open('mostbought.pkl','rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/choose',methods = ['POST'])
def choose():
    chosen = " "
    if request.method == 'POST':
        choice = request.form['Choose User or Item']
        if choice == 'User':
            chosen = 'userindex'
        else:
            chosen = "Item"
    return redirect(url_for(chosen))

@app.route('/userindex')
def userindex():
    return render_template('userindex.html')

@app.route('/Item')
def Item():
    return render_template('itemindex.html')

@app.route('/itemsubmit', methods = ['POST','GET'])
def itemsubmit():
    if request.method == 'POST':
        itemname = request.form['Items']
        number = int(request.form['number'])
        item_norm_sparse = csr_matrix(model.item_embeddings)
        similarities = cosine_similarity(item_norm_sparse)
        item_embed_dist_matrix = pd.DataFrame(similarities)
        item_embed_dist_matrix.columns = custitem.columns
        item_embed_dist_matrix.index = custitem.columns
        recommend_items = list(pd.Series(item_embed_dist_matrix.loc[itemname,:].sort_values(ascending = False).head(number+1).index[1:number+1]))
        recommend_items
        recitems = ", ".join(recommend_items)
    return render_template('userresult.html', output = recitems)

@app.route('/submit',methods = ['POST','GET'])
def submit():
    if request.method == 'POST':
        user = request.form['user']
        items = int(request.form['items'])
        result = " "

        if user in custdic.keys():
            n_cust,n_items = custitem.shape
            cust_x = custdic[user]
            scores = pd.Series(model.predict(cust_x,np.arange(n_items)))
            scores.index = custitem.columns
    
            scores = list(pd.Series(scores.sort_values(ascending = False).index))
    
            known_items = list(pd.Series(custitem.loc[user,:][custitem.loc[user,:] > 0].index).sort_values(ascending=False))
            scores = [x for x in scores if x not in known_items]
            return_score_list = scores[0:items]
            known_items = list(pd.Series(known_items))
            scores = list(pd.Series(return_score_list))
            result = "Hey items you may like ..."+" "+", ".join(scores)
        else:
            x = mostbought[:items]
            result ='Welcome!! the most bought items are'+" "+ ", ".join(x)
    return render_template('userresult.html',output = result)


if __name__ == '__main__':
    app.run(debug=False)