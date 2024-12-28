from _typeshed import Incomplete

CONTENT_TYPE_FORM_URLENCODED: str
CONTENT_TYPE_MULTI_PART: str

class ClientAuth:
    SIGNATURE_METHODS: Incomplete
    @classmethod
    def register_signature_method(cls, name, sign) -> None: ...
    client_id: Incomplete
    client_secret: Incomplete
    token: Incomplete
    token_secret: Incomplete
    redirect_uri: Incomplete
    signature_method: Incomplete
    signature_type: Incomplete
    rsa_key: Incomplete
    verifier: Incomplete
    realm: Incomplete
    force_include_body: Incomplete
    def __init__(
        self,
        client_id,
        client_secret: Incomplete | None = None,
        token: Incomplete | None = None,
        token_secret: Incomplete | None = None,
        redirect_uri: Incomplete | None = None,
        rsa_key: Incomplete | None = None,
        verifier: Incomplete | None = None,
        signature_method="HMAC-SHA1",
        signature_type="HEADER",
        realm: Incomplete | None = None,
        force_include_body: bool = False,
    ) -> None: ...
    def get_oauth_signature(self, method, uri, headers, body): ...
    def get_oauth_params(self, nonce, timestamp): ...
    def sign(self, method, uri, headers, body): ...
    def prepare(self, method, uri, headers, body): ...

def generate_nonce(): ...
def generate_timestamp(): ...