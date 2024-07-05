## High quality summarization

This repo contains de code to deploy a **[streamlit application](https://streamlit.io/)** that allows users to upload text documents and create high quality summarization using a generative AI model.

### Deployment

- Clone the repo to your local environment.
- Make sure that you have access to a generativeAI model:
   - In this app we are using [Amazon bedrock](https://aws.amazon.com/bedrock) via [langchain](https://python.langchain.com/v0.2/docs/introduction/).
    - If you want to use **Bedrock** check that you are authenticated in the AWS CLI. (You can see [AWS CLI configuration](https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-configure.html))
   - If you prefer to use a different model provider: change the _llm_chat_2_ object in the **high_quality.py** (line 11) and **regular-py** (line 70) files. Instead of using the BedrockChat from langchain choose the one you prefer. [Langchain providers](https://python.langchain.com/v0.2/docs/integrations/platforms/).
- Cd to the repo folder and Run `pip install -r requirements.txt`
- run `streamlit run streamlit_app.py`
