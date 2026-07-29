from src.core.attacker import canonical_output_model_spec
from src.core.models import CloudflareClient, get_client


def test_gemma_output_identity_is_account_neutral():
    expected = "cloudflare:@cf/google/gemma-4-26b-a4b-it"

    assert canonical_output_model_spec(
        "cloudflare:@cf/google/gemma-4-26b-a4b-it"
    ) == expected
    assert canonical_output_model_spec(
        "cloudflare2:@cf/google/gemma-4-26b-a4b-it"
    ) == expected


def test_other_secondary_models_keep_their_existing_output_identity():
    model_spec = "cloudflare2:@cf/moonshotai/kimi-k2.6"

    assert canonical_output_model_spec(model_spec) == model_spec


def test_cloudflare2_can_use_primary_credentials(monkeypatch):
    monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "primary-token")
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "primary-account")
    monkeypatch.setenv("CLOUDFLARE_API_TOKEN_2", "secondary-token")
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID_2", "secondary-account")

    client = get_client(
        "cloudflare2",
        model_name="@cf/google/gemma-4-26b-a4b-it",
        cloudflare_credentials="primary",
    )

    assert isinstance(client, CloudflareClient)
    assert client.profile_name == "primary"
    assert client.api_key == "primary-token"
    assert client.account_id == "primary-account"


def test_cloudflare2_keeps_secondary_credentials_by_default(monkeypatch):
    monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "primary-token")
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "primary-account")
    monkeypatch.setenv("CLOUDFLARE_API_TOKEN_2", "secondary-token")
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID_2", "secondary-account")

    client = get_client(
        "cloudflare2",
        model_name="@cf/google/gemma-4-26b-a4b-it",
    )

    assert isinstance(client, CloudflareClient)
    assert client.profile_name == "secondary"
    assert client.api_key == "secondary-token"
    assert client.account_id == "secondary-account"
