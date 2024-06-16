# Brian Lesko  12/3/2023
# helper library for GUI customization

import streamlit as st

# Version 1.0.0

class gui:
    def __init__(self):
        pass
    def about(self, photo = 'https://avatars.githubusercontent.com/u/116655452?v=4', author = "Brian", text = "In this code we ... "):
        with st.sidebar:
            col1, col2, = st.columns([1,5], gap="medium")
            with col1:
                st.image(photo)
            with col2:
                st.write(f""" 
                Hey it's {author}, \n
                {text}
                """)
            col1, col2, col3, col4, col5, col6 = st.columns([1,1,1,1,1,1.2], gap="medium")
            with col2: # Twitter
                st.write("[![X](https://raw.githubusercontent.com/BrianLesko/BrianLesko/main/.socials/svg-grey/x.svg)](https://twitter.com/BrianJosephLeko)")
            with col3: # GITHUB
                st.write("[![Github](https://raw.githubusercontent.com/BrianLesko/BrianLesko/main/.socials/svg-grey/github.svg)](https://github.com/BrianLesko)")
            with col4: # LINKEDIN
                st.write("[![LinkedIn](https://raw.githubusercontent.com/BrianLesko/BrianLesko/main/.socials/svg-grey/linkedin.svg)](https://www.linkedin.com/in/brianlesko/)")
            with col5: # YOUTUBE
                "." #st.write("[![LinkedIn](https://raw.githubusercontent.com/BrianLesko/BrianLesko/f7be693250033b9d28c2224c9c1042bb6859bfe9/.socials/svg-335095-blue/yt-logo-blue.svg)](https://www.linkedin.com/in/brianlesko/)")
            with col6: # BLOG Visual Study Code
                "." #"[![VSC]()](https://www.visualstudycode.com/)"

    def clean_format(self, wide=False):
        if wide == True: st.set_page_config(layout='wide')
        hide_st_style = """
                <style>
                #MainMenu {visibility: hidden;}
                footer {visibility: hidden;}
                header {visibility: hidden;}
                #root > div:nth-child(1) > div > div > div > div > section > div {padding-top: 0rem;}
                </style>
                """
        st.markdown(hide_st_style, unsafe_allow_html=True)

    def display_existing_messages(self,state):
        for msg in state.messages:
            st.chat_message(msg["role"]).write(msg["content"])

    def quick_setup(self,wide=False, text="In this code we ... "):
        self.clean_format(wide)
        self.about(text)