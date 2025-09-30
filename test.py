from langchain_core.prompts import PromptTemplate
from llm.llm import llm

# 1.创建提示词模板
prompt = PromptTemplate.from_template(template = "{foo}{bar}")

# 2.实时输入部分
partial_prompt1 = prompt.partial(foo="foo")			

# 3.最终提示词
print(partial_prompt1.format())

print(llm.invoke(foo='xx'))
