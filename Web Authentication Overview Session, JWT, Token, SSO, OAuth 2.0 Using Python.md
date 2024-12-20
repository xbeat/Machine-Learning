## Web Authentication Overview Session, JWT, Token, SSO, OAuth 2.0 Using Python
Slide 1: Web Authentication Overview

Web authentication is a crucial aspect of modern web applications, ensuring secure access to user accounts and protected resources. This slideshow explores various authentication methods, their implementations, and best practices.

```python
import hashlib

def hash_password(password):
    # Simple password hashing using SHA-256
    return hashlib.sha256(password.encode()).hexdigest()

# Example usage
user_password = "securePassword123"
hashed_password = hash_password(user_password)
print(f"Original password: {user_password}")
print(f"Hashed password: {hashed_password}")
```

Slide 2: WWW-Authenticate

WWW-Authenticate is a basic HTTP authentication mechanism. It prompts users for credentials when accessing protected resources.

```python
from flask import Flask, request, Response

app = Flask(__name__)

@app.route('/protected')
def protected():
    auth = request.authorization
    if not auth or not check_auth(auth.username, auth.password):
        return authenticate()
    return f"Hello {auth.username}!"

def authenticate():
    return Response('Could not verify your access level for that URL.\n'
                    'You have to login with proper credentials', 401,
                    {'WWW-Authenticate': 'Basic realm="Login Required"'})

def check_auth(username, password):
    return username == 'admin' and password == 'secret'

if __name__ == '__main__':
    app.run()
```

Slide 3: Session-based Authentication

Session-based authentication uses server-side storage to maintain user state across requests. It's widely used but has limitations for scalability and mobile apps.

```python
from flask import Flask, session, redirect, url_for, request
from flask_session import Session

app = Flask(__name__)
app.config['SESSION_TYPE'] = 'filesystem'
Session(app)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username == 'user' and password == 'password':
            session['logged_in'] = True
            return redirect(url_for('protected'))
    return '''
        <form method="post">
            <p><input type=text name=username>
            <p><input type=password name=password>
            <p><input type=submit value=Login>
        </form>
    '''

@app.route('/protected')
def protected():
    if 'logged_in' in session:
        return 'You are logged in!'
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run()
```

Slide 4: Token-based Authentication

Token-based authentication uses a signed token to verify user identity, reducing server-side storage needs and supporting mobile apps.

```python
import jwt
from flask import Flask, request, jsonify

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'

@app.route('/login', methods=['POST'])
def login():
    username = request.json.get('username', None)
    password = request.json.get('password', None)
    if username == 'user' and password == 'password':
        token = jwt.encode({'user': username}, app.config['SECRET_KEY'], algorithm='HS256')
        return jsonify({'token': token})
    return jsonify({'error': 'Invalid credentials'}), 401

@app.route('/protected')
def protected():
    token = request.headers.get('Authorization')
    if not token:
        return jsonify({'error': 'Token is missing'}), 401
    try:
        data = jwt.decode(token, app.config['SECRET_KEY'], algorithms=['HS256'])
        return jsonify({'message': f'Hello, {data["user"]}!'})
    except:
        return jsonify({'error': 'Token is invalid'}), 401

if __name__ == '__main__':
    app.run()
```

Slide 5: JSON Web Tokens (JWT)

JWTs are a compact, URL-safe means of representing claims to be transferred between two parties, commonly used for authentication and information exchange.

```python
import jwt
from datetime import datetime, timedelta

# JWT creation
def create_jwt(username):
    payload = {
        'sub': username,
        'iat': datetime.utcnow(),
        'exp': datetime.utcnow() + timedelta(minutes=30)
    }
    secret = 'your-secret-key'
    token = jwt.encode(payload, secret, algorithm='HS256')
    return token

# JWT verification
def verify_jwt(token):
    try:
        secret = 'your-secret-key'
        payload = jwt.decode(token, secret, algorithms=['HS256'])
        return payload
    except jwt.ExpiredSignatureError:
        return "Token has expired"
    except jwt.InvalidTokenError:
        return "Invalid token"

# Example usage
token = create_jwt('user123')
print(f"Created JWT: {token}")
print(f"Verified JWT: {verify_jwt(token)}")
```

Slide 6: Single Sign-On (SSO)

SSO allows users to access multiple applications with a single set of credentials, enhancing user experience and simplifying access management.

```python
from flask import Flask, request, redirect
from flask_oauthlib.client import OAuth

app = Flask(__name__)
oauth = OAuth(app)

google = oauth.remote_app(
    'google',
    consumer_key='your-google-client-id',
    consumer_secret='your-google-client-secret',
    request_token_params={
        'scope': 'email'
    },
    base_url='https://www.googleapis.com/oauth2/v1/',
    request_token_url=None,
    access_token_method='POST',
    access_token_url='https://accounts.google.com/o/oauth2/token',
    authorize_url='https://accounts.google.com/o/oauth2/auth',
)

@app.route('/')
def index():
    return 'Welcome! <a href="/login">Login with Google</a>'

@app.route('/login')
def login():
    return google.authorize(callback=url_for('authorized', _external=True))

@app.route('/logout')
def logout():
    session.pop('google_token', None)
    return redirect(url_for('index'))

@app.route('/login/authorized')
def authorized():
    resp = google.authorized_response()
    if resp is None:
        return 'Access denied: reason=%s error=%s' % (
            request.args['error_reason'],
            request.args['error_description']
        )
    session['google_token'] = (resp['access_token'], '')
    user_info = google.get('userinfo')
    return 'Logged in as: ' + user_info.data['email']

@google.tokengetter
def get_google_oauth_token():
    return session.get('google_token')

if __name__ == '__main__':
    app.run()
```

Slide 7: OAuth 2.0

OAuth 2.0 is an authorization framework that enables applications to obtain limited access to user accounts on an HTTP service.

```python
from requests_oauthlib import OAuth2Session
from flask import Flask, request, redirect, session, url_for
from flask.json import jsonify
import os

app = Flask(__name__)

# This information is obtained upon registration of a new GitHub OAuth
# application here: https://github.com/settings/applications/new
client_id = "<your-github-client-id>"
client_secret = "<your-github-client-secret>"
authorization_base_url = 'https://github.com/login/oauth/authorize'
token_url = 'https://github.com/login/oauth/access_token'

@app.route("/")
def demo():
    """Step 1: User Authorization.

    Redirect the user/resource owner to the OAuth provider (i.e. Github)
    using an URL with a few key OAuth parameters.
    """
    github = OAuth2Session(client_id)
    authorization_url, state = github.authorization_url(authorization_base_url)

    # State is used to prevent CSRF, keep this for later.
    session['oauth_state'] = state
    return redirect(authorization_url)

@app.route("/callback", methods=["GET"])
def callback():
    """ Step 3: Retrieving an access token.

    The user has been redirected back from the provider to your registered
    callback URL. With this redirection comes an authorization code included
    in the redirect URL. We will use that to obtain an access token.
    """

    github = OAuth2Session(client_id, state=session['oauth_state'])
    token = github.fetch_token(token_url, client_secret=client_secret,
                               authorization_response=request.url)

    # At this point you can fetch protected resources but lets save
    # the token and show how this is done from a persisted token
    # in /profile.
    session['oauth_token'] = token

    return redirect(url_for('.profile'))

@app.route("/profile", methods=["GET"])
def profile():
    """Fetching a protected resource using an OAuth 2 token.
    """
    github = OAuth2Session(client_id, token=session['oauth_token'])
    return jsonify(github.get('https://api.github.com/user').json())

if __name__ == "__main__":
    # This allows us to use a plain HTTP callback
    os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = "1"

    app.secret_key = os.urandom(24)
    app.run(debug=True)
```

Slide 8: Real-life Example: E-commerce Platform Authentication

Consider an e-commerce platform that implements token-based authentication for its mobile app and web interface.

```python
import jwt
from flask import Flask, request, jsonify
from functools import wraps

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'

def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.args.get('token')
        if not token:
            return jsonify({'message': 'Token is missing!'}), 403
        try:
            data = jwt.decode(token, app.config['SECRET_KEY'], algorithms=["HS256"])
        except:
            return jsonify({'message': 'Token is invalid!'}), 403
        return f(*args, **kwargs)
    return decorated

@app.route('/login')
def login():
    auth = request.authorization
    if auth and auth.password == 'password':
        token = jwt.encode({'user': auth.username}, app.config['SECRET_KEY'], algorithm="HS256")
        return jsonify({'token': token})
    return jsonify({'message': 'Could not verify!'}), 401

@app.route('/protected')
@token_required
def protected():
    return jsonify({'message': 'This is only available for authenticated users.'})

if __name__ == '__main__':
    app.run(debug=True)
```

Slide 9: Real-life Example: Multi-factor Authentication

Implementing multi-factor authentication for a sensitive application like a health records system.

```python
import pyotp
from flask import Flask, request, jsonify

app = Flask(__name__)

# In a real application, these would be stored securely, not as global variables
users = {'alice': {'password': 'password123', 'totp_secret': pyotp.random_base32()}}

@app.route('/login', methods=['POST'])
def login():
    username = request.json.get('username')
    password = request.json.get('password')
    totp_code = request.json.get('totp_code')

    if username not in users:
        return jsonify({'message': 'User not found'}), 404

    user = users[username]
    if user['password'] != password:
        return jsonify({'message': 'Invalid password'}), 401

    totp = pyotp.TOTP(user['totp_secret'])
    if not totp.verify(totp_code):
        return jsonify({'message': 'Invalid TOTP code'}), 401

    return jsonify({'message': 'Login successful'})

@app.route('/setup_2fa', methods=['POST'])
def setup_2fa():
    username = request.json.get('username')
    if username not in users:
        return jsonify({'message': 'User not found'}), 404

    secret = users[username]['totp_secret']
    totp = pyotp.TOTP(secret)
    provisioning_uri = totp.provisioning_uri(username, issuer_name="HealthApp")

    return jsonify({
        'secret': secret,
        'provisioning_uri': provisioning_uri
    })

if __name__ == '__main__':
    app.run(debug=True)
```

Slide 10: Security Considerations

When implementing authentication systems, consider these security aspects:

```python
import bcrypt
import secrets

def secure_password_hash(password):
    # Generate a salt and hash the password
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
    return hashed

def verify_password(stored_hash, provided_password):
    # Check if the provided password matches the stored hash
    return bcrypt.checkpw(provided_password.encode('utf-8'), stored_hash)

def generate_secure_token():
    # Generate a cryptographically secure token
    return secrets.token_urlsafe(32)

# Example usage
password = "user_password"
hashed_password = secure_password_hash(password)
print(f"Hashed password: {hashed_password}")

is_valid = verify_password(hashed_password, password)
print(f"Password verification result: {is_valid}")

secure_token = generate_secure_token()
print(f"Secure token: {secure_token}")
```

Slide 11: Best Practices for Authentication

Implementing robust authentication requires adherence to best practices:

```python
import re
from passlib.hash import argon2

def is_strong_password(password):
    # Check if password meets complexity requirements
    if (len(password) < 12 or
        not re.search("[a-z]", password) or
        not re.search("[A-Z]", password) or
        not re.search("[0-9]", password) or
        not re.search("[!@#$%^&*(),.?\":{}|<>]", password)):
        return False
    return True

def hash_password(password):
    # Use Argon2 for password hashing (considered more secure than bcrypt)
    return argon2.hash(password)

def verify_password(stored_hash, provided_password):
    # Verify the provided password against the stored hash
    return argon2.verify(provided_password, stored_hash)

# Example usage
password = "StrongP@ssw0rd123"
if is_strong_password(password):
    hashed = hash_password(password)
    print(f"Password meets complexity requirements. Hashed: {hashed}")
    
    # Simulating password verification
    is_valid = verify_password(hashed, password)
    print(f"Password verification result: {is_valid}")
else:
    print("Password does not meet complexity requirements")
```

Slide 12: Future Trends in Authentication

Exploring emerging trends in authentication technologies:

```python
import random
import string
from cryptography.fernet import Fernet

def simulate_biometric_auth():
    # Simulated biometric data (e.g., fingerprint hash)
    return ''.join(random.choices(string.ascii_letters + string.digits, k=64))

def passwordless_auth(user_id, biometric_data):
    # Simulate server-side verification
    stored_biometric = get_stored_biometric(user_id)
    return biometric_data == stored_biometric

def get_stored_biometric(user_id):
    # In a real system, this would retrieve from a secure database
    return simulate_biometric_auth()

# Example usage
user_id = "user123"
biometric_data = simulate_biometric_auth()
is_authenticated = passwordless_auth(user_id, biometric_data)

print(f"User authenticated: {is_authenticated}")
```

Slide 13: Continuous Authentication

Implementing continuous authentication to enhance security:

```python
import time
import random

class ContinuousAuthSystem:
    def __init__(self, user_id):
        self.user_id = user_id
        self.trust_score = 100  # Initial trust score
        self.last_action_time = time.time()
    
    def update_trust_score(self):
        current_time = time.time()
        time_diff = current_time - self.last_action_time
        
        # Decrease trust score over time
        self.trust_score -= time_diff * 0.1
        self.trust_score = max(0, min(100, self.trust_score))
        
        self.last_action_time = current_time
    
    def perform_action(self, action_type):
        self.update_trust_score()
        
        if action_type == "normal":
            self.trust_score += 5
        elif action_type == "suspicious":
            self.trust_score -= 10
        
        self.trust_score = max(0, min(100, self.trust_score))
        
        return self.trust_score > 50  # Action allowed if trust score > 50

# Example usage
auth_system = ContinuousAuthSystem("user123")

for _ in range(5):
    time.sleep(random.uniform(0.5, 2))  # Simulate time passing
    action_type = random.choice(["normal", "suspicious"])
    allowed = auth_system.perform_action(action_type)
    print(f"Action type: {action_type}, Allowed: {allowed}, Trust score: {auth_system.trust_score:.2f}")
```

Slide 14: Quantum-Resistant Cryptography

Preparing for post-quantum cryptography in authentication systems:

```python
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding, rsa

def generate_quantum_resistant_keypair():
    # Generate a 4096-bit RSA key (considered quantum-resistant for now)
    private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=4096
    )
    public_key = private_key.public_key()
    return private_key, public_key

def sign_message(message, private_key):
    signature = private_key.sign(
        message,
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()),
            salt_length=padding.PSS.MAX_LENGTH
        ),
        hashes.SHA256()
    )
    return signature

def verify_signature(message, signature, public_key):
    try:
        public_key.verify(
            signature,
            message,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )
        return True
    except:
        return False

# Example usage
private_key, public_key = generate_quantum_resistant_keypair()
message = b"Authenticate this message"
signature = sign_message(message, private_key)
is_valid = verify_signature(message, signature, public_key)

print(f"Signature valid: {is_valid}")
```

Slide 15: Additional Resources

For further exploration of authentication mechanisms and best practices:

1. OWASP Authentication Cheat Sheet: [https://cheatsheetseries.owasp.org/cheatsheets/Authentication\_Cheat\_Sheet.html](https://cheatsheetseries.owasp.org/cheatsheets/Authentication_Cheat_Sheet.html)
2. NIST Digital Identity Guidelines: [https://pages.nist.gov/800-63-3/](https://pages.nist.gov/800-63-3/)
3. OAuth 2.0 Specification: [https://oauth.net/2/](https://oauth.net/2/)
4. JWT RFC: [https://tools.ietf.org/html/rfc7519](https://tools.ietf.org/html/rfc7519)
5. Post-Quantum Cryptography: NIST's Post-Quantum Cryptography Standardization: [https://csrc.nist.gov/projects/post-quantum-cryptography](https://csrc.nist.gov/projects/post-quantum-cryptography)

## Response:
undefined

