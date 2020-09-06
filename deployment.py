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
    st.title('Parkinsons Prediction')
    html_temp="""
    <div style="background-color:orange; padding:15px;">
    <h2 style="text-align:center;font-family:verdana; font-size:300%; color:black;">PREDICTION BY INPUTS</h2>
    </div>
    <div >
    <p style="color:red;font-family:aerial">FOR STATUS 1  YOU HAVE DISEASE AND FOR 0 YOU ARE SAFE</p>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    
    l=[]
    for i in range(0,22):
        l.append(st.text_input(k[i],"Type Here"))
        
    infected_html="""
    <div style="background-color:red;padding:12px;">
    <h3 style="color:white;text-align:center">You are Infectected</h3>
    </div>
    """
    
    notinfected_html="""
    <div style="background-color:blue;padding:12px;">
    <h3 style="color:white;text-align:center">You are Infectected</h3>
    </div>
    """
    
    if st.button("Check Disease"):
        output = prediction(l)
        st.success("The patient's status is :{}".format(output))
        
        if output==1:
            st.markdown(infected_html,unsafe_allow_html=True)
        else:
            st.markdown(notinfected_html, unsafe_allow_html=True)
            
if __name__=='__main__':
    main()
