import os
from time import sleep
from typing import Callable, Any

import openai
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from pysentimiento import create_analyzer
from deep_translator import GoogleTranslator

# API Key
load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')


def chatgpt(message: str) -> str:
    """Obtiene respuesta de ChatGPT."""
    completion = openai.ChatCompletion.create(
        model='gpt-3.5-turbo-16k',
        messages=[
            {
                'role': 'user',
                'content': message
            }
        ],
        temperature=0.0
    )
    return completion.choices[0].message.content


def double_try(func: Callable[..., Any], *args: str) -> Any:
    """Intenta ejecutar el código varias veces."""
    # noinspection PyBroadException
    try:
        return func(*args)

    except:
        print(' ╠═══ <!> First retry. (Sleep 5)')
        sleep(5)

        # noinspection PyBroadException
        try:
            return func(*args)

        except:
            print(' ╠════ <!> Second and Last retry. (Sleep 8)')
            sleep(8)

            try:
                return func(*args)

            except Exception as e:
                print(' ╠═════ <!> <!> An error occurred:\n <!> <!> ', str(e))
                raise e


def double_try_v2(row_data: str, base_prompt: str) -> Any:
    """Try executing the code multiple times."""
    # Generate the full prompt using row_data
    full_prompt = base_prompt + row_data

    # noinspection PyBroadException
    try:
        return chatgpt(full_prompt)

    except:
        print(' ╠═══ <!> First retry. (Sleep 5)')
        sleep(5)

        # noinspection PyBroadException
        try:
            return chatgpt(full_prompt)

        except:
            print(' ╠════ <!> Second and Last retry. (Sleep 8)')
            sleep(8)

            try:
                return chatgpt(full_prompt)

            except Exception as e:
                print(' ╠═════ <!> <!> An error occurred:\n <!> <!> ', str(e))
                raise e


def process_survey(path: str) -> pd.DataFrame:
    """Clean survey data."""
    df = pd.read_csv(path)
    df = df.drop(['section_sis_id', 'section_id', 'sis_id'], axis=1)

    columns = [i for i in df.columns]
    for i in range(len(columns)):
        if ':' in columns[i]:
            df[' '.join(columns[i].split(' ')[1:])] = df.iloc[:, i]

    df = df.drop(columns=columns[5:])
    columns = [i for i in df.columns]
    for i in columns:
        if df[i].isna().sum() == df.shape[0]:
            df = df.drop(i, axis=1)
    return df


def traduccion(x: str) -> str:
    """Traduce textos al inglés"""
    if isinstance(x, str):
        x = GoogleTranslator(source='auto', target='en').translate(x)
    return x


def language(lang: str) -> str:
    """Map acronym with full language."""
    dicc = {
        'ENG': 'English',
        'ESP': 'Spanish',
        'POR': 'Portuguese'
    }
    return dicc[lang]


analyzer = create_analyzer(task='sentiment', lang='en')


def sent_an(testimonio: str) -> str:
    """Get sentiment from English comments."""
    # Set default value
    sentiment = 'Neutral'
    if isinstance(testimonio, str):
        try:
            sent = analyzer.predict(testimonio)

            # Results
            sentiment_dic = {
                'POS': 'Positive',
                'NEG': 'Negative',
                'NEU': 'Neutral'
            }
            sentiment = sentiment_dic[sent.output]
        except RuntimeError:
            prompt = f"""You are a sentiment analysis model.
                     You identify whether a comment was positive, neutral
                     or negative. The input is a single comment, so your output
                     should only be either `Positive`, `Neutral` or `Negative`.
                     Do not add anything else.

                     Comment to be analyzed:
                     {testimonio}
                     """
            sentiment = double_try(chatgpt, prompt)
    return sentiment


analyzer_esp = create_analyzer(task='sentiment', lang='es')


def sent_an_esp(testimonio: str) -> str:
    """Get sentiment from Spanish comments."""
    sent = analyzer_esp.predict(testimonio)

    # Results
    sentiment = {
        'POS': 'Positive',
        'NEG': 'Negative',
        'NEU': 'Neutral'
    }
    return sentiment[sent.output]


def join_strings(group):
    return group.astype(str).agg('||'.join)
