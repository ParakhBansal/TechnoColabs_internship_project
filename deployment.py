import streamlit as st
import pickle
import numpy as np
import pandas as pd

model =  pickle.load(open('xgbmodel.pkl','rb'))

df = pd.read_csv('parkinsons.data')
x = df.drop(['name','status'],axis=1)

k = np.array(x.columns)

def prediction(values):
    input = np.array([values]).astype(np.float64)
    pred = model.predict(input)
    return pred

def main():
    
    page_bg_img = '''
    <style>
    body {
    background-image: url("https://i.pinimg.com/736x/93/16/c2/9316c2b8665e3bc2e35ae6cb5f45d533.jpg");
    background-size: cover;
    } 
    </style>
    '''
    
    st.markdown(page_bg_img, unsafe_allow_html=True)
    
    
    html_temp="""
    
    <div style="background-image:url('https://wpamelia.com/wp-content/uploads/2019/02/astronomy-constellation-dark-998641.jpg');height=200px; width:80vw;margin-bottom:10px">
    <h2 style="text-align:center;font-family:verdana; size:100px; color:white;">PARKINSON'S PREDICTIONS</h2>
    </div>
    <div>
    <p style="color:black;size:70px;font-family:aerial">FOR STATUS 1  YOU HAVE DISEASE AND FOR 0 YOU ARE SAFE</p>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    
    l=[]
    for i in range(0,22):
        l.append(st.text_input(k[i]))
        
    infected_html="""
    <div style="background-color:red;padding:12px;">
    <h3 style="color:white;text-align:center">You are Infectected</h3>
    </div>
    """
    
    notinfected_html="""
    <div style="background-color:blue;padding:12px;">
    <h3 style="color:white;text-align:center">You are not Infectected</h3>
    </div>
    """
    
    if st.button("Check Disease"):
        count =0
        for j in range(0,22):
            if not l[j]:
                count = 1
        if count == 0:
            output = prediction(l)
            st.success("The patient's status is :{}".format(output))
            if output==1:
                 st.markdown(infected_html,unsafe_allow_html=True)
                 st.warning('oh no!! you are suffering from parkinsons please seek medical guidance')
            else:
                 st.markdown(notinfected_html, unsafe_allow_html=True)
                 st.balloons()
        else:
            st.error('all fields are not filled kindly fill all the data for prediction')
if __name__=='__main__':
    main()
