from datetime import datetime, timedelta, timezone
import ipaddress

from cryptography import x509
from cryptography.x509.oid import NameOID, ExtendedKeyUsageOID
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa


def make_localhost_cert(
    cert_path="localhost.crt",
    key_path="localhost.key",
    days_valid=365,
):
    # 1) Private key
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)

    # 2) Subject / Issuer (self-signed => same)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "localhost")])

    now = datetime.now(timezone.utc)

    # 3) SAN (important for browsers)
    san = x509.SubjectAlternativeName([
        x509.DNSName("localhost"),
        x509.IPAddress(ipaddress.ip_address("127.0.0.1")),
        x509.IPAddress(ipaddress.ip_address("::1")),
    ])

    # 4) Build cert
    cert = (
        x509.CertificateBuilder()
            .subject_name(name)
            .issuer_name(name)
            .public_key(key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(now - timedelta(minutes=1))
            .not_valid_after(now + timedelta(days=days_valid))
            .add_extension(san, critical=False)
            .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
            .add_extension(
                x509.KeyUsage(
                    digital_signature=True,
                    key_encipherment=True,
                    content_commitment=False,
                    data_encipherment=False,
                    key_agreement=False,
                    key_cert_sign=False,
                    crl_sign=False,
                    encipher_only=False,
                    decipher_only=False,
                ),
                critical=True,
            )
            .add_extension(
                x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]),
                critical=False,
            )
            .sign(private_key=key, algorithm=hashes.SHA256())
    )

    # 5) Write key (PEM)
    with open(key_path, "wb") as f:
        f.write(
            key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.TraditionalOpenSSL,  # "RSA PRIVATE KEY"
                encryption_algorithm=serialization.NoEncryption(),
            )
        )

    # 6) Write cert (PEM)
    with open(cert_path, "wb") as f:
        f.write(
            cert.public_bytes(
                serialization.Encoding.PEM
            )
        )

    return cert_path, key_path


if __name__ == "__main__":
    crt, key = make_localhost_cert()
    print("Generated:", crt, key)
