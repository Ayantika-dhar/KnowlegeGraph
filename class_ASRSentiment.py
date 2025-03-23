class ASRSentiment(torch.nn.Module):
    # Wrapper for automatic speech recognition, followed by GNN-based graph embedding
    def __init__(self, tiny_asr=False):
        super(ASRSentiment, self).__init__()
        self.asr_model = FastASR(use_tiny_model=tiny_asr)

    def to_device(self, device):
        if device == 'cuda' and not torch.cuda.is_available():
            print('CUDA not available.')
        elif device not in ('cpu', 'cuda'):
            print('Device can only be cpu or cuda.')
        else:
            self.asr_model.pipe.model.to(device)

    def process_video(self, input_tensor=None, video_path=None, sr=None, **kwargs):
        from KG.graph_embedding_pipeline import text_to_graph_embedding

        output = self.asr_model.process_video(input_tensor, video_path, sr=sr)
        text = output['text'].strip()

        # Language logic preserved
        languages = list(dict.fromkeys([chunk['language'] for chunk in output['chunks']]))
        language = languages[0] if len(languages) == 1 else None

        # Prepare response if no text found
        if text == '':
            return {'features': [], 'predictions': [], 'language': None, 'asr': ''}

        # Use unified pipeline to get graph embedding
        graph_embedding = text_to_graph_embedding(text)  # shape: (1, 768)

        return {
            'features': graph_embedding.detach(),
            'predictions': [],
            'language': language,
            'asr': text
        }
