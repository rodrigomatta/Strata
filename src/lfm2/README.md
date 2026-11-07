# LFM2-MoE local (experimental)

Este alvo recebe texto ou IDs de tokens e gera texto em uma GPU AMD com HIP.
O tokenizer lê o vocabulário, as regras BPE e os IDs BOS/EOS do GGUF.
O caminho do Qwen fica separado. `STRATA_LFM2` é `OFF` por padrão.

## GUI integrada (Linux, HIP gfx1100)

A partir da raiz do repositório, com as dependências de `apps/host` já instaladas:

```sh
npm --prefix apps/host run dev:lfm2 -- --model /path/to/LFM2.5-24B-A2B-Q4_0.gguf
```

Esse comando compila a engine e abre os servidores da GUI em
`http://localhost:5199`. O GGUF é carregado automaticamente, sem copiar ou
baixar pesos. O padrão da integração é **32768 tokens de contexto**, sem
limite fixo de tokens de saída. A geração termina em EOS, ao pressionar Parar
ou quando o contexto fica cheio. Prompt, histórico, raciocínio e resposta
compartilham essa janela. Para outro tamanho, use `--context N`.
Não inicie uma segunda cópia enquanto a primeira ainda ocupa a GPU.

Em Modelos, o cartão LFM2 permite escolher o caminho do GGUF, carregar e
descarregar o modelo. O chat usa o seletor de modelos da GUI, mantém o
histórico e mostra raciocínio separado da resposta. O menu de raciocínio
oferece ligado/desligado; não há níveis ou orçamento de raciocínio nessa engine.
O botão Parar cancela a geração e mantém os pesos para a próxima mensagem.

O host usa um processo HIP persistente (`--worker`) e uma API OpenAI Chat
privada em loopback, com chave aleatória por carga. A API pública do host
encaminha `/v1/models` e `/v1/chat/completions`; o chat da GUI usa
`/api/chat/stream`. Requisições são enfileiradas, uma de cada vez. O padrão de
amostragem é greedy; temperatura e top-k continuam disponíveis. Sem
`max_tokens`, ou com `max_tokens: 0`, a API usa todo o espaço restante do contexto.
Um limite positivo explícito na API é respeitado. No chat da GUI, o limite
fixo de saída fica desligado, inclusive se havia um valor salvo anteriormente.

Só conversa de texto está integrada: imagens, ferramentas/agentes, saída
estruturada, Responses e Anthropic não estão implementados para o LFM2.
O servidor Python `serve/` e o comando `strata launch` permanecem separados.
O instalador desktop ainda não distribui o binário LFM2; este fluxo usa o checkout.

### Validação da integração em 22/10/2026

Na RX 7900 XT (gfx1100) com Xeon E5-2697 v3 e este GGUF:

- Builds HIP isolado e pela raiz, referência CPU e CTest 1/1 passaram.
- TypeScript e build de produção da GUI passaram. O Vite avisa sobre o tamanho
  de alguns bundles; o build concluiu.
- 11 testes do adaptador, controle nativo e gateway passaram.
- A rodada final das rotas de chat, controles nativos e raciocínio teve
  31 testes aprovados, zero falhas e um teste do Harness pulado por falta
  desse componente. Log: `/tmp/lfm2-final-targeted-tests.log`.
- Na GPU com contexto 32768, duas requisições greedy produziram a mesma saída
  sem recarregar os pesos. Histórico, UTF-8, streaming, raciocínio separado,
  EOS, amostragem, recusa de contexto excedido e cancelamento passaram.
- O chat greedy do café mediu 93,23 tok/s de decode; com temperatura 0,7 e
  top-k 40, 92,52 tok/s. Pesos: 13,57 GB de VRAM, além dos buffers de contexto.
- Na API do host, uma requisição sem `max_tokens` gerou 1103 tokens antes do
  cancelamento do teste. A próxima mensagem pela rota da GUI concluiu em EOS,
  com `contextMax: 32768` e uso de tokens informado pela engine.
- O chat foi operado no navegador: respondeu em português e mostrou o
  raciocínio recolhido e estatísticas reais.

Suíte completa do host: 979 testes, 882 passaram, 96 foram pulados e um falhou.
A falha é `timeskip-real.test.mjs`, ao verificar se o processo neto terminou.
Ela também ocorreu em uma cópia limpa de `011c9e60`, sem esta integração.
Não foi corrigida como parte do LFM2.

Não foram testados outros modelos, outras GPUs/backends, o instalador Tauri
ou um prompt que preencha todos os 32768 tokens. A janela foi alocada em 32k;
isso não é uma medição de qualidade ou velocidade em contexto cheio.

Logs: `/tmp/lfm2-gui-gpu-32k.log`, `/tmp/lfm2-gui-host-32k.log`,
`/tmp/lfm2-integration-tests.log`, `/tmp/lfm2-host-full-tests.log` e
`/tmp/lfm2-timeskip-baseline.log`.
Os testes reais podem ser repetidos com `tools/test_gui_gpu.mjs` (carrega a
GPU) ou `tools/test_gui_host.mjs` (usa uma GUI já rodando); não rode os dois
servidores de modelo na GPU ao mesmo tempo.

## Compilar

Build isolado, validado na RX 7900 XT (gfx1100), em Linux:

```sh
cmake -S src/lfm2 -B ./build-lfm2 \
  -DCMAKE_HIP_ARCHITECTURES=gfx1100 -DCMAKE_BUILD_TYPE=Release
cmake --build ./build-lfm2 -j2
ctest --test-dir ./build-lfm2 --output-on-failure
```

Isso cria `strata-lfm2`, a referência de CPU `lfm2_ref` e o teste unitário.
`-DSTRATA_LFM2_REF=OFF` desliga a referência.
No build isolado, `-DBUILD_TESTING=OFF` desliga o teste.

Também é possível adicionar os alvos ao build da raiz com
`-DSTRATA_LFM2=ON -DCMAKE_HIP_ARCHITECTURES=gfx1100`.
Nesse caso, os testes seguem `STRATA_BUILD_TESTS`.
O build da raiz foi configurado e os dois executáveis foram compilados com
`STRATA_NATIVE_EXPERTS=OFF` e `STRATA_BUILD_TESTS=OFF`, sem baixar dependências.

## Usar

Exemplos a partir da raiz do repositório:

```sh
MODEL=/path/to/LFM2.5-24B-A2B-Q4_0.gguf

# Só tokenizar: não inicializa a GPU nem carrega os pesos nela.
./build-lfm2/strata-lfm2 --tok "$MODEL" "Olá, café!"
./build-lfm2/lfm2_ref --tok "$MODEL" "Olá, café!" --no-bos

# Continuar um texto, sem aplicar um template de conversa.
./build-lfm2/strata-lfm2 "$MODEL" -p "Once upon a time" -n 64

# Um turno ChatML.
./build-lfm2/strata-lfm2 "$MODEL" -p "Responda em português: o que é uma GPU?" --chat -n 512

# Amostragem opcional. Sem --temp, a escolha é greedy.
./build-lfm2/strata-lfm2 "$MODEL" -p "Conte uma história curta." --chat -n 512 --temp 0.7 --top-k 40

# Interface antiga: número de tokens a gerar, seguido pelos IDs do prompt.
./build-lfm2/strata-lfm2 "$MODEL" 8 124894 35808
```

O formato de `--chat` é
`<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n`.
O BOS é incluído conforme o GGUF. `--no-bos` vale apenas para `--tok`.
O texto sai aos poucos em stdout. Diagnóstico e velocidades saem em stderr.
A geração para no EOS do GGUF, em `<|im_end|>`, em `<|endoftext|>` ou no limite
de `-n`. Os marcadores `<think>` e `</think>` aparecem no texto.
`--chat` é um turno; ainda não há conversa interativa com histórico.

`LFM2_CTX` define o contexto alocado, com padrão 4096. Prompt e limite de
geração precisam caber nele. `--top-k 0` usa todo o vocabulário na amostragem;
com temperatura zero, `--top-k` não muda a escolha greedy.

## Comparar o tokenizer

Use Python com `gguf` e `tokenizers`, um `tokenizer.json` de referência
independente e o mesmo GGUF. O script verifica a identidade do vocabulário,
os cinco textos pedidos e, com `--fuzz 100`, mais 100 textos mistos.
Cada texto é comparado com e sem a política de BOS do GGUF.

```sh
python3 src/lfm2/tools/compare_tok.py \
  --binary ./build-lfm2/strata-lfm2 \
  --gguf "$MODEL" --tokenizer /tmp/lfm2-tokenizer.json --fuzz 100
```

O script não baixa arquivos e não executa operações na GPU.
Os arquivos de referência em `/tmp` não fazem parte do repositório.

## Validação local anterior

Hardware: AMD Radeon RX 7900 XT (gfx1100, 21,46 GB informados pelo HIP) e
Intel Xeon E5-2697 v3. A referência usou 14 threads OpenMP.

Passou:

- Build isolado dos três alvos e CTest: 1/1.
- Build dos dois executáveis pela raiz com `STRATA_LFM2=ON`.
- Configuração pela raiz com `STRATA_LFM2=OFF`: nenhum alvo LFM2 aparece.
- Com testes da raiz desligados, o subdiretório não cria `BUILD_TESTING`.
- Comparação do tokenizer do executável HIP: 210 comparações, zero falhas,
  incluindo inglês, português com acentos, números, contrações e ChatML.
  Os 125017 IDs reais coincidiram com a referência; 2983 entradas são padding.
  O log anterior da referência de CPU registra 60 comparações, zero falhas.
- `--tok "Hello"`: `124894 35808`, sem inicializar a GPU.
- Testes unitários com ASan/UBSan: sem erro. A detecção de leaks foi desligada.
  Os caracteres de substituição impressos pelo teste são intencionais:
  ele envia bytes inválidos e um caractere incompleto ao stream.
- Logs anteriores de GPU e CPU pelo modo de IDs: oito passos com stdout
  idêntico, incluindo top-5 e logits impressos com três casas decimais.
- O prefixo do chat gerado anteriormente pela GPU coincide com a nova saída
  da CPU. Isso compara texto; os tokens do chat não foram registrados.
- EOS na CPU: `-p "Responda apenas com a palavra: café" --chat -n 512`
  emitiu 82 tokens e parou em `124900 (<|im_end|>)`. A saída terminou em
  `</think>café`.
- Amostragem na CPU: `-p "Hello" -n 8 --temp 0.7 --top-k 40` concluiu com
  oito tokens. Os testes unitários também verificam a restrição de top-k.

Falhou:

- Nenhum build, teste unitário ou comparação concluída apresentou falha.

Interrompido / não concluído:

- A nova execução de EOS na GPU foi encerrada com SIGTERM durante a carga
  dos pesos. O `amd-smi process` mostrou um `llama-server` existente usando
  14,7 GB de VRAM; o teste LFM2 tinha conseguido alocar 4,2 GB. A engine precisa
  de 13,57 GB só para os pesos. O servidor existente foi mantido em execução.
- Naquela rodada, EOS e amostragem ficaram pendentes na GPU; foram concluídos
  na validação com GPU liberada abaixo. O teste unitário cobre também
  `<|endoftext|>`, que não foi observado em geração real.

Medições registradas, com esse GGUF e o hardware acima:

| Execução | Prefill | Decode |
| --- | --- | --- |
| GPU, IDs, log anterior com cache quente | 2 tokens, 83,76 tok/s | 7 passos, 84,74 tok/s |
| GPU, chat do café, log anterior | 17 tokens, 87,89 tok/s | 63 passos, 89,50 tok/s |
| CPU, novo chat do café até EOS, 14 threads | 17 tokens, 2,16 tok/s | 82 passos, 2,71 tok/s |

Na primeira linha, a carga levou 2,95 s e ocupou 13,57 GB de VRAM em pesos.
O chat da GPU emitiu 64 tokens e atingiu o limite ainda dentro de `<think>`.
Decode conta os passos após o prefill; o primeiro token é escolhido a partir
do prefill. Esses são testes curtos, não uma comparação com outra engine.
As medições da GPU não foram repetidas com sucesso naquela retomada.
Os logs locais estão em `/tmp/lfm2-validation/` e não fazem parte do Git.

Arquivos da alteração local antes da integração da GUI: `CMakeLists.txt` da raiz (+6 linhas) e, neste
diretório, `CMakeLists.txt`, `lfm2_engine.hip`, `lfm2_ref.cpp`, `lfm2_model.hpp`,
`lfm2_gguf.hpp`, `lfm2_vocab.hpp`, `lfm2_tokenizer.hpp`,
`lfm2_unicode_data.inc`, `lfm2_cli.hpp`, `tests/unit.cpp`,
`tools/compare_tok.py`, `THIRD_PARTY_NOTICES` e este `README.md`.
Antes da integração, o diff fora daqui continha apenas a opção CMake e seu bloco condicionado.
Não foi feita comparação binária do Qwen com a última release.

## Teste após atualizar o Strata

Em 21/10/2026, a base foi atualizada de `9de04cd4` para `011c9e60`
(`v0.2.2`, engine 0.1.47) por merge fast-forward de `origin/main`.
A alteração local do CMake foi reaplicada sem conflitos. Os checksums dos
13 arquivos locais do LFM2 coincidiram antes de acrescentar esta seção.
Um backup ficou em `/tmp/strata-update-backup-20261021/`.

Passaram após o merge: build isolado em `/tmp/lfm2-postmerge`, CTest 1/1,
build dos executáveis pela raiz, ausência de alvos LFM2 com a opção desligada,
210 comparações do tokenizer sem falhas e o teste `--tok Hello` nos
executáveis HIP e CPU. O chat na CPU emitiu os mesmos 82 tokens de antes,
com stdout idêntico, e parou em `124900 (<|im_end|>)`.
Com 14 threads, mediu 3,80 tok/s no prefill e 3,68 tok/s no decode.
O teste de amostragem na CPU, com temperatura 0,7 e top-k 40, também concluiu.
Essas medições não isolam o efeito da atualização: a carga da máquina varia.

Nenhum teste concluído falhou. A geração na GPU não foi executada nesta
validação, por escolha de manter o `llama-server` existente em funcionamento.
O binário HIP foi compilado e seu modo de tokenização foi testado sem usar
a GPU. A suíte geral do Strata e outros backends não foram executados.
Os logs desta rodada estão em `/tmp/lfm2-validation/post-merge/`.

## Validação com a GPU liberada

Ainda na base `011c9e60` (Strata 0.2.2), foi feito um build Release novo em
`build-lfm2/`, para HIP gfx1100, com os executáveis `strata-lfm2`, `lfm2_ref`
e o teste unitário. O CTest passou 1/1. Os testes abaixo foram executados
na RX 7900 XT com o Xeon E5-2697 v3 e o mesmo GGUF de 24B.

| Teste GPU | Prefill | Decode | Parada |
| --- | --- | --- | --- |
| Chat greedy, café | 17 tokens, 91,69 tok/s | 82 passos, 93,71 tok/s | EOS 124900 |
| Mesmo chat, temperatura 0,7 e top-k 40 | 17 tokens, 94,97 tok/s | 129 passos, 93,11 tok/s | EOS 124900 |
| Modo por IDs | 2 tokens, 75,57 tok/s | 7 passos, 89,19 tok/s | limite de 8 tokens |
| Texto sem ChatML, `-p Hello` | 2 tokens, 74,65 tok/s | 7 passos, 93,62 tok/s | limite de 8 tokens |

O chat greedy produziu stdout idêntico ao chat da CPU registrado após o
merge: 82 tokens e saída terminando em `</think>café`. A amostragem também
terminou em `café`, com UTF-8 válido. Os marcadores de raciocínio continuam
visíveis. O modo por IDs foi comparado com uma nova execução da CPU com
14 threads: texto, top-5 e logits impressos coincidiram em todos os oito
passos. O texto sem ChatML coincidiu com a continuação do modo por IDs.

Os pesos ocuparam 13,57 GB de VRAM. A primeira carga levou 18,04 s; nas três
execuções seguintes, de 2,77 a 3,14 s. As velocidades são de execuções curtas;
não representam um teste de contexto longo ou carga concorrente.

Nenhum teste desta rodada falhou. `<|endoftext|>` continua coberto apenas
pelo teste unitário; a parada real observada foi `<|im_end|>`.
A suíte geral do Strata, outros modelos e outros backends não foram rodados.
Logs e verificações: `/tmp/lfm2-validation/gpu-final/`, incluindo `checks.json`.

Para executar o binário compilado, a partir da raiz do repositório:

```sh
./build-lfm2/strata-lfm2 \
  /path/to/LFM2.5-24B-A2B-Q4_0.gguf \
  -p "Olá! Responda em português." --chat -n 512
```

## Limites

Só foi testado o `LFM2.5-24B-A2B-Q4_0.gguf`, com pesos Q4_0 e Q6_K,
em Linux/HIP na gfx1100. A engine mantém todos os pesos na GPU.
A GUI local do host está integrada conforme descrito acima. O servidor Python
`serve/` e o comando `strata` não foram adaptados.
CUDA, SYCL, Metal, Windows e outras GPUs não foram compilados nem testados.
A referência de CPU foi escrita neste diretório; a comparação GPU/CPU
verifica a concordância entre elas, não com uma engine externa.
