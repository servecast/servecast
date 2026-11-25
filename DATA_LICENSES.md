# Data licences

ServeCast ships the request lengths of three public traces, under `src/servecast/data/traces/`. Only the input and
output token counts of each request are kept. Each file names its source, licence and our changes; they are repeated
here.

| file | source | licence | our changes |
|---|---|---|---|
| `azure-conversation.json` | Azure Public Dataset, Azure LLM inference trace 2023, conversation (Patel et al., 2023), https://github.com/Azure/AzurePublicDataset | CC-BY-4.0 | input and output token counts only; requests whose prompt and output exceed 4,096 tokens removed |
| `burstgpt.json` | BurstGPT v2.0, without failures (Wang et al., KDD 2025), https://github.com/HPMLL/BurstGPT | CC-BY-4.0 | input and output token counts only; requests whose prompt and output exceed 4,096 tokens removed; every 27th remaining request kept, 50,000 in all |
| `servegen.json` | ServeGen, offline trace d, text requests (Alibaba Cloud Model Studio), https://github.com/alibaba/ServeGen | Apache-2.0 | input and output token counts only; requests whose prompt and output exceed 4,096 tokens removed |

The workload laws under `src/servecast/data/workload_laws/` are fitted from these three files and carry the same
licences.
