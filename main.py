import streamlit as st
import pandas as pd
#from bokeh.models.widgets import Div

def main():
    st.title('Similarity Recommender')
    st.markdown("---")
    st.text("This is a lead generator according to a company's portfolio.")

    st.header('header')
    st.subheader('subheader')

    st.markdown('loading files')
    file = st.file_uploader('choose your file', type='csv')
    if file is not None:
        df = pd.read_csv(file)
        st.dataframe(df.head(5))

    st.sidebar.title("Useful Links")
    st.sidebar.markdown("---")
    #st.sidebar.markdown("<hr>", unsafe_allow_html=True)
    #st.sidebar.image("assets/codenation.png", use_column_width=True)
    st.sidebar.markdown("[![Github]"
                        "(https://www.startpage.com/av/proxy-image?piurl=https%3A%2F%2Fcdn.iconscout.com%2Ficon%2Ffree%2Fpng-256%2Fgithub-153-675523.png&sp=1594759674Tdf76077b6f2588b1077c86da4bf33f55adb5d35e49be7104e1150f33fceb117a)]"
                        "(https://github.com/Rpinto02)")
    st.sidebar.markdown("[![Linkedin]"
                         "(https://www.startpage.com/av/proxy-image?piurl=https%3A%2F%2Fcdn.iconscout.com%2Ficon%2Ffree%2Fpng-256%2Flinkedin-42-151143.png&sp=1594758987Ta3a7ba5e5bc165c95644e199516c6fc7a4a136a143d412c97997fa27bd624989)]"
                         "(https://www.linkedin.com/in/rpinto02/)")
    st.sidebar.markdown("[![Codenation]"
                        "(<img src='file://codenation.png' alt='alt text' width='200'/>)]"
                        "(https://codenation.dev)")








if __name__ == '__main__':
    main()