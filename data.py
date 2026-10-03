import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

# Function to process uploaded file
def process_uploaded_file(data):
    if data is not None:
        file_extension = data.name.split(".")[-1].lower()
        supported_formats = ["csv", "xlsx", "xls"]
        if file_extension in supported_formats:
            if file_extension == "csv":
                data_df = pd.read_csv(data)
            elif file_extension in ["xlsx", "xls"]:
                data_df = pd.read_excel(data)
        else:
            raise ValueError("Unsupported file format. Please upload a CSV or Excel file.")
        return data_df
    else:
        return None

# Page Maintenance Status
status = False

# Setting Page Configuration
st.set_page_config(page_title="ExploreData Tool", page_icon="icon3.png", layout="wide")

# Sidebar
st.sidebar.title("ExploreData Tool")
st.sidebar.caption("Developed by Vinay")
st.sidebar.write("""
**ExploreData** is designed to make data analysis easy and accessible for everyone.
Upload your dataset and explore a variety of insights and visualizations.

**Features:**
- Upload datasets in CSV, XLSX, or XLS formats
- View basic insights about your data
- Generate statistics
- Create visualizations
- Display code snippets for reproducibility
""")

# Main Page
st.title("ExploreData: Your Interactive Data Analysis Tool")
st.write("""
Welcome to ExploreData, your one-stop solution for performing interactive exploratory data analysis (EDA) on any dataset in (.csv/.xlsx/.xls) format. Whether you're a data enthusiast, analyst, or researcher, this web application empowers you to gain valuable insights from your data effortlessly.
""")

if status:
    imag = Image.open("mn.png")
    st.image(imag, use_column_width=True)
    st.title("Sorry, the web app is currently under maintenance. Please try again later.")
else:
    file1 = st.file_uploader("Upload a DataSet in (.csv/.xlsx/.xls) Format")
    if file1 is not None:
        data = process_uploaded_file(file1)
        if data is not None:
            st.write("Preview:")
            st.dataframe(data)

            # Basic Insights
            show_basic_insights = st.checkbox('Show Basic Insights💡')
            if show_basic_insights:
                st.title("Basic Insights💡")
                if st.checkbox("How many rows and columns in the DataSet?"):
                    st.write("Number of rows : ", len(data))
                    st.write("Number Of Columns : ", len(data.columns))
                    if st.button('Show Code for Rows and Columns'):
                        st.code("""print("Number of rows : ", len(data))\nprint("Number Of Columns : ", len(data.columns))""")
                if st.checkbox("Display the dimension and Shape of DataSet"):
                    st.write("Dimension : ", data.ndim)
                    st.write("Shape : ", data.shape)
                    if st.button('Show Code for Dimension and Shape'):
                        st.code("""print("Dimension : ", data.ndim)\nprint("Shape : ", data.shape)""")
                if st.checkbox("List the name of Attributes/columns in dataset"):
                    st.write("Columns : ")
                    st.write(pd.DataFrame(data.columns, columns=["Attribute Names"]))
                    if st.button('Show Code for Columns'):
                        st.code("""print("Columns : ")\nprint(pd.DataFrame(data.columns, columns=["Attribute Names"]))""")
                if st.checkbox("Display the count of Non-Null values in dataset"):
                    st.write(data.count())
                    if st.button('Show Code for Non-Null Count'):
                        st.code("""print(data.count())""")
                if st.checkbox("Display the count of Null values in dataset"):
                    st.write(data.isna().sum())
                    if st.button('Show Code for Null Count'):
                        st.code("""print(data.isna().sum())""")
                if st.checkbox("Display the Data types of each column"):
                    st.write(pd.DataFrame(data.dtypes, columns=["data type"]))
                    if st.button('Show Code for Data Types'):
                        st.code("""print(pd.DataFrame(data.dtypes, columns=["data type"]))""")
                if st.checkbox("Display the Data of Numeric columns"):
                    num_data = data.select_dtypes(exclude=["object", "bool"])
                    st.dataframe(num_data)
                    if st.button('Show Code for Numeric Data'):
                        st.code("""num_data = data.select_dtypes(exclude=["object", "bool"])\nprint(num_data)""")
                if st.checkbox("Display the Data of Object Columns"):
                    obj_data = data.select_dtypes(include="object")
                    st.write(obj_data)
                    if st.button('Show Code for Object Data'):
                        st.code("""obj_data = data.select_dtypes(include="object")\nprint(obj_data)""")
                for i in data.select_dtypes(include="object"):
                    if st.checkbox(f"Display the Unique Values of '{i}' Column"):
                        st.write(data[i].unique())
                        if st.button(f'Show Code for Unique Values of {i}'):
                            st.code(f"print(data['{i}'].unique())")

            # Statistics
            show_statistics = st.checkbox('Show Statistics🔍')
            if show_statistics:
                st.title("STATISTICS🔍")
                num_data = data.select_dtypes(exclude=["object", "bool"])
                obj_data = data.select_dtypes(include="object")
                for i in num_data:
                    if st.checkbox(f"Display the Basic Statistics of '{i}' Column"):
                        st.write(data[i].describe())
                        if st.button(f'Show Code for Statistics of {i}'):
                            st.code(f"print(data['{i}'].describe())")
                for i in obj_data:
                    if st.checkbox(f"Display the Mode of '{i}' Column"):
                        st.write(data[i].mode(dropna=False))
                        if st.button(f'Show Code for Mode of {i}'):
                            st.code(f"print(data['{i}'].mode(dropna=False))")
                if st.checkbox("Display the statistics of Numeric data in dataset"):
                    st.write(data.describe())
                    if st.button('Show Code for Numeric Statistics'):
                        st.code("print(data.describe())")

            # Visualization
            show_visualization = st.checkbox('Show Visualization📊')
            if show_visualization:
                st.title("VISUALIZATION📊")
                ty = st.selectbox("Select Type ", ['Distribution', 'Correlation'])
                x_l = st.selectbox("On X - AXIS ", list(data.columns))
                if ty == 'Correlation':
                    y_l = st.selectbox("On Y - AXIS ", list(data.columns))
                if st.button("Plot Graph"):
                    fig, ax = plt.subplots(1, figsize=(20, 8))
                    if ty == 'Distribution':
                        plt.title(f"The {ty} of {x_l}")
                        sns.histplot(x=x_l, data=data, kde=True, ax=ax)
                        plt.xticks(rotation='vertical')
                        st.pyplot(fig)
                    else:
                        plt.title(f"The {ty} of {x_l} and {y_l}")
                        sns.scatterplot(x=x_l, y=y_l, data=data, ax=ax)
                        plt.xticks(rotation='vertical')
                        st.pyplot(fig)
                if st.button('Show Code for Plot'):
                    if ty == 'Distribution':
                        st.code(f"""fig, ax = plt.subplots()\nplt.title("The {ty} of {x_l}")\nsns.histplot(x='{x_l}', data=data, kde=True, ax=ax)\nplt.xticks(rotation='vertical')\nst.pyplot(fig)""")
                    else:
                        st.code(f"""fig, ax = plt.subplots()\nplt.title("The {ty} of {x_l} and {y_l}")\nsns.scatterplot(x='{x_l}', y='{y_l}', data=data, kde=True, ax=ax)\nplt.xticks(rotation='vertical')\nst.pyplot(fig)""")
