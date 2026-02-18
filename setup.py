from setuptools import setup, find_packages

setup(
    name="llm-cost-tracker",
    version="1.0.0",
    description="Track, analyse, and control LLM API spending across OpenAI, Anthropic, Google, Mistral and more",
    author="Prajit Datta",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[],
    extras_require={"dev": ["pytest>=7.0"]},
)
