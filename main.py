import streamlit as st
import pandas as pd

def main():
    st.title('Hello World!')
    st.header('header')
    st.subheader('subheader')
    st.text('texting')
    st.image('codenation.png')
    st.markdown('loading files')
    file = st.file_uploader('choose your file', type='csv')
    if file is not None:
        df = pd.read_csv(file)
        st.dataframe(df.head(5))



if __name__ == '__main__':
    main()