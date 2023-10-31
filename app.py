import re
import io

import openpyxl
import numpy as np

from survey_functions import *

# GA logo
_, col2, _ = st.columns(3)
with col2:
    st.image('https://globalalumni.org/wp-content/uploads/logo-global.png')

# Header and description
with st.container():
    st.title('EncuestApp')
    st.markdown('Aplicación para procesar la información de las encuestas')

# Select survey type
tipo = st.selectbox(
    'Selecciona tipo de encuesta',
    ('<Selecciona una opción>', 'INITIAL', 'MIDTERM', 'FINAL')
)

if '<' not in tipo:
    # Upload excel file
    archivo = st.file_uploader('Sube el archivo con los datos de encuestas:')

    # Check if uploaded and select sheet to read
    if archivo is not None:
        workbook = openpyxl.load_workbook(archivo, read_only=True)
        hojas = workbook.sheetnames
        hoja = st.selectbox(
            'Elige hoja del excel con los datos:',
            ['<Selecciona una opción>'] + hojas
        )

        # Show sample data
        if hoja != '<Selecciona una opción>':
            df = pd.read_excel(archivo, sheet_name=hoja)
            st.write(df.head())

            # Check program codes
            ga_code = 'Program'
            if ga_code not in df.columns:
                ga_code = st.selectbox(
                    'Códigos GA',
                    ['<Selecciona una opción>'] + list(df.columns)
                )

            # Check special colummn
            special = False
            if 'DPTM' in [i.split('-')[0] for i in df[ga_code].unique()]:
                special = True
                col1, col2, col3 = st.columns(3)

                # Select columns to process
                with col1:
                    questions = st.multiselect(
                        'Escoge columnas a procesar:',
                        list(df.columns),
                    )

                # Special
                with col2:
                    special_ops = st.multiselect(
                        'Elige preguntas para DPTM:',
                        list(df.columns),
                    )

                with col3:
                    name = st.selectbox(
                        'Identificador de alumnos (SSID):',
                        list(df.columns)
                    )

            else:
                # Select columns to process
                questions = st.multiselect(
                    'Escoge columnas a procesar:',
                    list(df.columns),
                )

            # Run app
            sleep(5)
            _, col2, _ = st.columns(3)
            with col2:
                button = st.button('Procesar encuestas')

            if button:
                with st.spinner('Procesando los datos...'):
                    # Extract language from course code
                    df['lang'] = [i.split('-')[1] for i in df[ga_code]]

                    # Translate testimonials
                    cond = ~df['lang'].isin(['ENG', 'ESP'])
                    eng = df['lang'] != 'ESP'
                    esp = df['lang'] == 'ESP'
                    df_sent = pd.DataFrame()
                    for question in questions:
                        emojis = ['📝', '📒', '📗'][np.random.randint(0, 3)]
                        st.write(f'{emojis} Traduciendo la columna: \n {question}')
                        df_sent[question] = df[question]
                        df[f'c_{question}'] = df[question].fillna('Neutral comment').astype(str)
                        df.loc[cond, f'c_{question}'] = df.loc[cond, f'c_{question}'].apply(traduccion)

                        # Get sentiment
                        emoji = ['😃', '😐', '😖'][np.random.randint(0, 3)]
                        st.write(f'{emoji} Obteniendo el sentimiento de: \n {question}')
                        df_sent[f's_{question}'] = None
                        df_sent.loc[eng, f's_{question}'] = df.loc[eng, f'c_{question}'].apply(sent_an)
                        df_sent.loc[esp, f's_{question}'] = df.loc[esp, f'c_{question}'].apply(sent_an_esp)
                        df = df.drop([f'c_{question}'], axis=1)
                    df = df.drop('lang', axis=1)

                    # Get summaries
                    result_df = df.groupby(ga_code)[questions].apply(join_strings).reset_index()
                    prompt_base = f"""You are a professional academic analyst.
                    Based on students' answers from a survey, you detect
                    incoveniences, nuances and difficulties they've experienced.
                    You are capable of identifying whether students liked the course,
                    found it of little help or were indifferent. Positive, neutral
                    and negative aspects are to be targeted.

                    The question students are answering is `{question}`.
                    Based on a batch of answers separated by `||`, summarize the general feelings,
                    focusing on the aforementioned points. It is not necessary to
                    use bulletpoints or explicitly mention which point you're referring to,
                    simply summarize the comments and state important aspects that made students
                    like or dislike the course they took.

                    Students' comments:
                    """
                    for question in questions:
                        emoji = ['📕', '📖', '📜'][np.random.randint(0, 3)]
                        st.write(f'{emoji} Resumiendo: \n {question}')
                        result_df[f's_{question}'] = result_df[question].apply(
                            double_try_v2,
                            args=(prompt_base,)
                        )
                        result_df = result_df.drop(question, axis=1)

                    # Identify complaints in 'DPTM'
                    if special:
                        df['codes'] = [i.split('-')[0] for i in df[ga_code]]
                        dpm_df = df.loc[df['codes'] == 'DPTM', [name, ga_code] + questions]
                        dpm_df = dpm_df.groupby(name)[special_ops].apply(join_strings).reset_index()

                        # Prompt engineering
                        prompt_base = f"""You are an expert academic analyst.
                        Your task is to identify which students' comments are
                        complaining about tight deadlines and the lack of extending a course.
                        This course has some mandatory deadlines that if not met, automatically
                        marks students with `Fail`. Also, it is not possible to extend the course
                        or ask for extra time to finish assigments. It's imperative that you
                        detect which comments are referring to these issues.

                        Thus, you should output `Yes` if the comment meets the aforementioned criteria,
                        and `No` otherwise. Do not explain your decision and do not add anything apart
                        from `Yes` or `No`.

                        Student's comment:
                        """
                        for op in special_ops:
                            emoji = ['😠', '😡', '🙃'][np.random.randint(0, 3)]
                            st.write(f'{emoji} Identificando quejas en: \n {op}')
                            dpm_df[op] = dpm_df[op].apply(
                                double_try_v2,
                                args=(prompt_base,)
                            )
                            dpm_df[op] = [re.sub(r'[\'"`]', '', i) for i in dpm_df[op]]
                        df = df.drop('codes', axis=1)

                    # Get results
                    _, col2, _ = st.columns(3)
                    with col2:
                        buffer = io.BytesIO()
                        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                            df_sent.to_excel(writer, sheet_name='Sentiment')
                            result_df.to_excel(writer, sheet_name='Summaries')
                            if special:
                                dpm_df.to_excel(writer, sheet_name='Complaints')
                            writer.close()

                            # Botón para descargar
                            st.download_button(
                                label='📥 Descargar datos',
                                data=buffer,
                                file_name=f'survey_analysis_{tipo.lower()}.xlsx',
                                mime='application/vnd.ms-excel'
                            )
