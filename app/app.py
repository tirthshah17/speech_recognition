import streamlit as st
from record import sound


def main():

    st.title('Oral Reading Assessment')

    if st.button('Record'):
        with st.spinner('Recording in progress.....'):
            sound.record_audio()

    if st.button('Stop'):
        sound.stop_recording()
        st.success('Recorded Successfully')

    


if __name__ == '__main__':
    main()