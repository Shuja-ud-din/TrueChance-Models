import asyncio
import time
from camel_tools.disambig.bert import BERTUnfactoredDisambiguator
from camel_tools.tagger.default import DefaultTagger
from camel_tools.tokenizers.word import simple_word_tokenize

# Separate semaphore for tashkeel
TASHKEEL_CONCURRENCY = 2
tashkeel_semaphore = asyncio.Semaphore(TASHKEEL_CONCURRENCY)

disambiguator = None
tagger = None


def load_tashkeel():
    global disambiguator, tagger

    print("🔤 Loading CAMeL BERT diacritizer...")
    disambiguator = BERTUnfactoredDisambiguator.pretrained(
        model_name='msa',
        use_gpu=True
    )
    tagger = DefaultTagger(disambiguator, 'diac')
    print("✅ Tashkeel model loaded.")


async def diacritize(text: str):
    async with tashkeel_semaphore:
        tokens = simple_word_tokenize(text)

        start = time.time()
        diacritized_tokens = await asyncio.to_thread(
            tagger.tag,
            tokens
        )

        result = " ".join(diacritized_tokens)
        latency = (time.time() - start) * 1000

    return result, latency
