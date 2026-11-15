"""Pacote principal do projeto.

Ao ser importado, faz o Python verificar conexões HTTPS com os certificados
do sistema operacional (pacote truststore). Sem isso, em máquinas com
antivírus ou proxy que inspecionam o tráfego HTTPS, as chamadas ao Azure
falham com erro de certificado. A verificação continua ativa: só muda a
lista de autoridades em que o Python confia.
"""

try:
    import truststore

    truststore.inject_into_ssl()
except ImportError:  # sem o pacote, vale o comportamento padrão do Python
    pass
