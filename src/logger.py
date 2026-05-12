DEBUG = True

def log(msg: str, level: str = 'INFO'):
    """
    Sistema de debug unificado    
    """
    if DEBUG:
        prefijos = {
            "INFO":    "🔵",
            "SUCCESS": "✅",
            "WARN":    "⚠️",
            "ERROR":   "❌",
            "QUERY":   "🔍",
            "SCORE":   "📊", 
        }
        prefijo = prefijos.get(level, "•")
        print(f"{prefijo} [{level}] {msg}")