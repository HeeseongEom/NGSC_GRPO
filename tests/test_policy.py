import torch

from ngsc_grpo.policy import LinearBetaController, analytic_reference_kl, reference_from_action


BOUNDS = {"eta": [0.0, 3.0], "tau": [0.0, 1.0], "gamma": [0.0, 1.0], "kappa_sp": [0.0, 4.0]}


def test_linear_beta_controller_has_96_parameters_and_reference_init():
    action = torch.tensor([1.2, 0.4, 0.7, 2.0])
    reference = reference_from_action(action, BOUNDS, concentration=20.0)
    policy = LinearBetaController()
    policy.initialize_as_reference(reference)
    assert sum(parameter.numel() for parameter in policy.parameters()) == 96
    states = torch.randn(5, 11)
    predicted = policy.mean_action(states, BOUNDS)
    assert torch.allclose(predicted, action.expand_as(predicted), atol=1e-5)
    assert torch.allclose(analytic_reference_kl(policy.distribution(states), reference), torch.zeros(5), atol=1e-5)
