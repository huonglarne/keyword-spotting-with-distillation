from src.teacher_precompute import get_pretrained_ast

def test_get_pretrained_ast():
    pretrained_mdl_path = 'checkpoints/speechcommands_10_10_0.9812.pth'    
    audio_model = get_pretrained_ast(pretrained_mdl_path)
    assert audio_model is not None