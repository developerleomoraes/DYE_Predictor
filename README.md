bbb# Guia Completo para Configuração de Ambiente Virtual Python

![Python Virtual Environment](https://img.shields.io/badge/Python-3.10%2B-blue)

## Índice
1. [Introdução](#introdução)
2. [Pré-requisitos](#pré-requisitos)
3. [Instalação do Ambiente Virtual](#instalação-do-ambiente-virtual)
4. [Ativação do Ambiente](#ativação-do-ambiente)
5. [Gerenciamento de Dependências](#gerenciamento-de-dependências)
6. [Chaves API's](#chaves-apis)

---

## Introdução

Ambientes virtuais são ferramentas essenciais para desenvolvimento Python, permitindo isolar dependências por projeto. Este guia cobre todo o processo de configuração usando `virtualenv` com Python 3.10.14+.

---

## Pré-requisitos

✅ Python 3.10.14 ou superior instalado  
✅ Pip atualizado  
✅ Acesso ao terminal/linha de comando  

**Verifique sua instalação:**
### Linux/macOS
python3 --version
python3 -m pip --version

### Windows
python --version
python -m pip --version



## Instalação do Ambiente Virtual

No seu terminal instale a biblioteca virtualenv com o seguinte comando `pip install virtualenv` ou a instalação referente no sistema Windows


## Ativação do Ambiente

No seu terminal, garanta estar na pasta raiz do diretório do projeto e digite o comando no ambiente linux: `source -m venv .venv` ou para o ambiente Windows: .`\nome_do_ambiente\Scripts\activate`

Após isso, check se o nome do ambiente aparece ao lado esquerdo do path do seu terminal.


## Gerenciamento de Dependências

Para a instalação das bibliotecas necessárias para rodar o projeto,
no seu terminal dentro agora do ambiente virtual digite o seguinte comando: `pip install -r requirements.txt`, e agora é apenas escolher o ambiente virtual para rodar o projeto em casos de arquivos .ipynb.


## Chaves API's

> Importante mencionar que é necessário criar um arquivo .env contendo as chaves API's das mesmas para execução do projeto
    As API's utilizada foram: </br>
        - https://www.alphavantage.co/ </br>
        - https://www.eia.gov/opendata/ </br>
        - https://fred.stlouisfed.org/docs/api/fred/

> Vale ressaltar que a documentação de como gerar as chaves API's gratuitas estão nos próprios sites.
