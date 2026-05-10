from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

def _estimate_tokens(text: str) -> int:
    if not text:
        return 0
    words = len(text.split())
    by_words = int(words * 1.3)
    by_chars = int(len(text) / 4)
    return max(1, max(by_words, by_chars))

@dataclass
class GroqLLMClient:
    api_key: str
    model: str = "llama-3.3-70b-versatile"
    temperature: float = 0.0
    max_new_tokens: int = 8192
    top_k: int = 40
    top_p: float = 0.95
    seed: int = 42

    def invoke(self, prompt: str, **kwargs) -> str:
        from langchain_groq import ChatGroq
        from langchain_core.messages import HumanMessage, AIMessage
        import time
        import random

        logger.debug(
            "📏 Prompt tokens(est.): %d | chars: %d | model: %s",
            _estimate_tokens(prompt),
            len(prompt or ""),
            self.model,
        )
        attempts_total = 3
        last_error: Exception | None = None
        for attempt in range(attempts_total):
            try:
                llm = ChatGroq(
                    api_key=self.api_key,
                    model=self.model,
                    temperature=self.temperature,
                    max_tokens=self.max_new_tokens,
                    model_kwargs={
                        "top_p": self.top_p,
                        "seed": self.seed,
                    }
                )
                response = llm.invoke([HumanMessage(content=prompt)])
                return response.content
            except Exception as e:
                last_error = e
                msg = str(e).lower()
                if "429" in msg or "rate limit" in msg or "ratelimit" in msg:
                    delay = 60.0
                else:
                    delay = (2 ** attempt) + random.uniform(0, 1)
                logger.warning(
                    "⏳ Erro na chamada Groq. Retrying in %.2fs | attempt %d/%d | model: %s | err: %s",
                    delay,
                    attempt + 1,
                    attempts_total,
                    self.model,
                    str(e),
                )
                time.sleep(delay)
        raise RuntimeError(str(last_error))
        
@dataclass
class OllamaLLMClient:
    base_url: str
    model: str
    temperature: float = 0.0
    max_new_tokens: int = 8192
    num_ctx: int = 16384
    top_k: int = 40
    top_p: float = 0.95
    seed: int = 42

    def invoke(self, prompt: str, **kwargs) -> str:
        from langchain_ollama.llms import OllamaLLM
        logger.debug(
            "📏 Prompt tokens(est.): %d | chars: %d | model: %s",
            _estimate_tokens(prompt),
            len(prompt or ""),
            self.model,
        )
        llm = OllamaLLM(model=self.model, temperature=self.temperature,
                        max_new_tokens=self.max_new_tokens, num_ctx=self.num_ctx,
                        base_url=self.base_url,
                        top_k=self.top_k, top_p=self.top_p, seed=self.seed)
        return llm.invoke(prompt)  

@dataclass
class OpenRouterLLMClient:
    api_key: str
    model: str
    base_url: str = "https://openrouter.ai/api/v1"
    temperature: float = 0.0
    max_new_tokens: int = 8192
    top_k: int = 1
    top_p: float = 1.0
    seed: int = 42
    frequencyPenalty: float = 0.0
    presencePenalty: float = 0.0
    repetitionPenalty: float =1.0
    retries: int = 2
    timeout: int = 60
    provider_preferences: dict = None

    def invoke(self, prompt: str, response_format: dict | None = None, **kwargs) -> str:
        import requests
        import random
        import time
        logger.debug(
            "📏 Prompt tokens(est.): %d | chars: %d | model: %s",
            _estimate_tokens(prompt),
            len(prompt or ""),
            self.model,
        )
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost",
            "X-Title": "fc_ai_pipeline",
        }
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "frequencyPenalty" : self.frequencyPenalty,
            "presencePenalty":self.presencePenalty,
            "repetitionPenalty": self.repetitionPenalty,
            "max_completion_tokens": self.max_new_tokens,
            "seed": self.seed,
            "stream": False
        }
        if response_format:
            payload["response_format"] = response_format
        if self.provider_preferences:
            payload["provider"] = self.provider_preferences

        last_error = None
        attempts_total = max(1, self.retries)
        for attempt in range(attempts_total):
            try:
                resp = requests.post(url, json=payload, headers=headers, timeout=self.timeout)
                if resp.status_code == 429:
                    retry_after = resp.headers.get("retry-after") or resp.headers.get("Retry-After")
                    delay = None
                    if retry_after:
                        try:
                            delay = float(retry_after)
                        except Exception:
                            delay = None
                    if delay is None:
                        delay = 60.0
                    else:
                        delay = max(60.0, delay)
                    last_error = f"HTTP 429 Rate limited. body={resp.text[:500]}"
                    if attempt == attempts_total - 1:
                        raise RuntimeError(last_error)
                    logger.warning(
                        "⏳ Rate limited (429). Retrying in %.2fs | attempt %d/%d | model: %s",
                        delay,
                        attempt + 1,
                        attempts_total,
                        self.model,
                    )
                    time.sleep(delay)
                    continue
                resp.raise_for_status()
                data = resp.json()
                if isinstance(data, dict):
                    if "choices" in data and data.get("choices"):
                        c = data["choices"][0]
                        if "message" in c and isinstance(c["message"], dict):
                            return c["message"].get("content", "")
                        if "text" in c:
                            return c["text"]
                    if "error" in data:
                        last_error = data.get("error")
                        time.sleep(1.0 * (attempt + 1))
                        continue
                return str(data)
            except Exception as e:
                last_error = e
                delay = None
                try:
                    if isinstance(e, requests.HTTPError) and getattr(e, "response", None) is not None:
                        if e.response.status_code == 429:
                            delay = 60.0
                except Exception:
                    delay = None
                if delay is None:
                    msg = str(e).lower()
                    if "429" in msg or "rate limit" in msg or "ratelimit" in msg:
                        delay = 60.0
                    else:
                        delay = (2 ** attempt) + random.uniform(0, 1)
                logger.warning(
                    "⏳ Erro na chamada OpenRouter. Retrying in %.2fs | attempt %d/%d | model: %s | err: %s",
                    delay,
                    attempt + 1,
                    attempts_total,
                    self.model,
                    str(e),
                )
                time.sleep(delay)
        raise RuntimeError(str(last_error))
        
def make_llms(cfg):
    if cfg.llm.provider == "ollama":
        mk = lambda m: OllamaLLMClient(cfg.llm.base_url, m, cfg.llm.params.temperature, cfg.llm.params.max_new_tokens, cfg.llm.params.num_ctx, top_k=cfg.llm.params.top_k, top_p=cfg.llm.params.top_p, seed=cfg.llm.params.seed)

    elif cfg.llm.provider == "groq":
        mk = lambda m: GroqLLMClient( api_key=cfg.llm.api_key, model=m, temperature=cfg.llm.params.temperature, max_new_tokens=cfg.llm.params.max_new_tokens, top_p=cfg.llm.params.top_p, seed=cfg.llm.params.seed)
    elif cfg.llm.provider == "openrouter":
        def mk(m):
            provider_params = getattr(cfg.llm, "provider_params", None) or {}
            provider_prefs = provider_params.get(m)
            if not provider_prefs:
                m_lower = m.lower()
                for k, v in provider_params.items():
                    if str(k).lower() in m_lower:
                        provider_prefs = v
                        break

            if provider_prefs and provider_prefs.get("quantizations"):
                logger.warning(
                    "⚙️ [CONFIG] Quantização aplicada para modelo '%s': %s",
                    m,
                    provider_prefs.get("quantizations"),
                )
            
            return OpenRouterLLMClient(
                api_key=cfg.llm.api_key, 
                model=m, 
                base_url=(cfg.llm.base_url or "https://openrouter.ai/api/v1"), 
                temperature=cfg.llm.params.temperature, 
                max_new_tokens=cfg.llm.params.max_new_tokens,  
                top_k=cfg.llm.params.top_k, 
                top_p=cfg.llm.params.top_p, 
                seed=cfg.llm.params.seed,
                provider_preferences=provider_prefs
            )
    
    else:
        raise ValueError(f"Provedor LLM desconhecido: {cfg.llm.provider}")
    
    return mk(cfg.llm.models.chat)