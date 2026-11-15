# Tech Challenge Fase 4 — Monitoramento multimodal de pacientes

Projeto da Fase 4 da pós-graduação IA para Devs (FIAP POSTECH).

O sistema analisa vídeo, áudio, texto e sinais vitais de um paciente, detecta anomalias em cada fonte,
combina os resultados em um nível de risco e emite um alerta para a equipe médica.

> Em construção. As instruções completas de instalação e execução serão adicionadas ao final do projeto.

## Estrutura

```
src/
  modelos.py     tipos compartilhados (Evento, Risco)
  video/         análise de postura e detecção de pessoas
  audio/         síntese, transcrição, análise de texto e de voz
  anomalias/     sinais vitais, prescrições e movimentação
  fusao/         cálculo do risco e emissão do alerta
tests/           testes automatizados (pytest)
notebooks/       exploração e resultados de cada etapa
```

## Testes

```bash
pytest -q
```
