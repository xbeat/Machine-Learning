## Django Two-Factor Authentication for Secure User Login
Slide 1: Django Two-Factor Authentication Setup

Django two-factor authentication enhances security by requiring users to provide additional verification beyond passwords. The django-two-factor-auth package integrates seamlessly with Django's authentication system, providing a robust foundation for implementing 2FA in web applications.

```python
# settings.py
INSTALLED_APPS = [
    'django.contrib.auth',
    'django_otp',
    'django_otp.plugins.otp_totp',
    'django_otp.plugins.otp_static',
    'two_factor',
]

MIDDLEWARE = [
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django_otp.middleware.OTPMiddleware',
]

# Enable two-factor auth
LOGIN_URL = 'two_factor:login'
LOGIN_REDIRECT_URL = 'two_factor:profile'
```

Slide 2: Basic TOTP Configuration

Time-based One-Time Password (TOTP) implementation requires proper configuration of secret key generation and verification mechanisms. This setup establishes the foundation for generating secure temporary codes compatible with authenticator apps.

```python
# models.py
from django_otp.plugins.otp_totp.models import TOTPDevice
from django.contrib.auth.models import User

def create_totp_device(user: User, name: str = "default"):
    device = TOTPDevice.objects.create(
        user=user,
        name=name,
        confirmed=True
    )
    return device.config_url
```

Slide 3: Custom Authentication Backend

A custom authentication backend enables fine-grained control over the two-factor authentication process, allowing for additional validation steps and integration with existing user management systems.

```python
# backends.py
from django.contrib.auth.backends import ModelBackend
from django_otp import user_has_device, devices_for_user

class TwoFactorAuthBackend(ModelBackend):
    def authenticate(self, request, username=None, password=None, **kwargs):
        user = super().authenticate(request, username=password, **kwargs)
        if user and user_has_device(user):
            devices = devices_for_user(user)
            for device in devices:
                if device.verify_token(kwargs.get('otp_token')):
                    return user
        return None
```

Slide 4: User Registration with 2FA

The registration process must be modified to include two-factor authentication setup. This implementation guides users through device registration and initial verification while maintaining security standards.

```python
# views.py
from django.contrib.auth.decorators import login_required
from django_otp.plugins.otp_totp.models import TOTPDevice
import qrcode
import io

@login_required
def setup_2fa(request):
    if request.method == 'POST':
        device = TOTPDevice.objects.create(
            user=request.user,
            name='default',
            confirmed=False
        )
        
        # Generate QR code
        qr = qrcode.QRCode(version=1, box_size=10, border=5)
        qr.add_data(device.config_url)
        qr.make(fit=True)
        
        img = qr.make_image(fill_color="black", back_color="white")
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        qr_code = buffer.getvalue()
        
        return JsonResponse({'qr_code': base64.b64encode(qr_code).decode()})
```

Slide 5: Custom Middleware for 2FA Enforcement

A specialized middleware ensures that protected routes require two-factor authentication completion, redirecting users to appropriate setup or verification pages when necessary.

```python
# middleware.py
from django.shortcuts import redirect
from django.urls import reverse
from django_otp import user_has_device

class Require2FAMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response
        
    def __call__(self, request):
        if request.user.is_authenticated:
            if not user_has_device(request.user):
                if request.path != reverse('setup_2fa'):
                    return redirect('setup_2fa')
        
        response = self.get_response(request)
        return response
```

Slide 6: API Token Generation with 2FA Verification

The API token generation process must incorporate two-factor authentication verification to maintain security standards when issuing access tokens. This implementation ensures tokens are only generated after successful 2FA completion.

```python
# utils.py
from django.core.signing import TimestampSigner
from django_otp import devices_for_user
import secrets

def generate_api_token(user, otp_token):
    devices = devices_for_user(user)
    for device in devices:
        if device.verify_token(otp_token):
            signer = TimestampSigner()
            token = secrets.token_urlsafe(32)
            return signer.sign(token)
    return None

# views.py
@require_POST
def get_api_token(request):
    otp_token = request.POST.get('otp_token')
    token = generate_api_token(request.user, otp_token)
    if token:
        return JsonResponse({'token': token})
    return JsonResponse({'error': 'Invalid 2FA token'}, status=401)
```

Slide 7: Backup Codes Implementation

Backup codes provide a fallback authentication method when primary 2FA devices are unavailable. This implementation generates, stores, and validates single-use backup codes securely.

```python
# models.py
from django.db import models
import secrets

class BackupCodes(models.Model):
    user = models.ForeignKey('auth.User', on_delete=models.CASCADE)
    code = models.CharField(max_length=8)
    used = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)

    @classmethod
    def generate_codes(cls, user, count=10):
        codes = []
        for _ in range(count):
            code = secrets.token_hex(4)
            backup = cls.objects.create(user=user, code=code)
            codes.append(code)
        return codes
```

Slide 8: Recovery Process Integration

The recovery process handles scenarios where users lose access to their 2FA devices, implementing a secure method to verify identity and restore account access using backup codes or alternative verification methods.

```python
# views.py
from django.contrib.auth import login
from .models import BackupCodes

class RecoveryView(FormView):
    template_name = 'recovery.html'
    form_class = RecoveryForm
    
    def form_valid(self, form):
        user = form.get_user()
        backup_code = form.cleaned_data['backup_code']
        
        valid_backup = BackupCodes.objects.filter(
            user=user,
            code=backup_code,
            used=False
        ).first()
        
        if valid_backup:
            valid_backup.used = True
            valid_backup.save()
            login(self.request, user)
            return redirect('two_factor:setup')
            
        return self.form_invalid(form)
```

Slide 9: Rate Limiting and Security Measures

Implementation of rate limiting and security measures prevents brute force attacks and ensures the 2FA system remains secure under various threat scenarios.

```python
# decorators.py
from django.core.cache import cache
from functools import wraps
from django.http import HttpResponseForbidden

def rate_limit_2fa(max_attempts=5, timeout=300):
    def decorator(view_func):
        @wraps(view_func)
        def wrapped(request, *args, **kwargs):
            ip = request.META.get('REMOTE_ADDR')
            cache_key = f'2fa_attempts_{ip}'
            attempts = cache.get(cache_key, 0)
            
            if attempts >= max_attempts:
                return HttpResponseForbidden('Too many attempts')
            
            cache.set(cache_key, attempts + 1, timeout)
            return view_func(request, *args, **kwargs)
        return wrapped
    return decorator
```

Slide 10: SMS-Based Two-Factor Authentication

SMS-based two-factor authentication provides an alternative verification method, implementing secure message delivery and verification through mobile phone numbers.

```python
# services.py
from twilio.rest import Client
from django.conf import settings
import random

class SMSVerification:
    def __init__(self):
        self.client = Client(
            settings.TWILIO_ACCOUNT_SID,
            settings.TWILIO_AUTH_TOKEN
        )
    
    def send_verification_code(self, phone_number):
        code = str(random.randint(100000, 999999))
        self.client.messages.create(
            body=f'Your verification code is: {code}',
            from_=settings.TWILIO_PHONE_NUMBER,
            to=phone_number
        )
        return code
```

Slide 11: Device Management Interface

A comprehensive device management system allows users to view, add, and remove registered 2FA devices while maintaining security protocols and audit trails.

```python
# views.py
from django.views.generic import ListView
from django_otp import devices_for_user

class DeviceManagementView(ListView):
    template_name = 'device_management.html'
    context_object_name = 'devices'
    
    def get_queryset(self):
        return devices_for_user(self.request.user)
    
    def post(self, request, *args, **kwargs):
        action = request.POST.get('action')
        device_id = request.POST.get('device_id')
        
        if action == 'remove':
            device = self.get_queryset().filter(id=device_id).first()
            if device and not device.is_primary:
                device.delete()
                
        return self.get(request, *args, **kwargs)
```

Slide 12: Security Audit Logging

Implementation of comprehensive security audit logging tracks all 2FA-related activities, enabling security monitoring and compliance reporting while maintaining detailed records of authentication attempts and device management.

```python
# models.py
from django.db import models
from django.contrib.auth.models import User

class SecurityAuditLog(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    timestamp = models.DateTimeField(auto_now_add=True)
    action = models.CharField(max_length=50)
    ip_address = models.GenericIPAddressField()
    device_id = models.CharField(max_length=50, null=True)
    success = models.BooleanField()
    
    @classmethod
    def log_2fa_attempt(cls, user, request, success, device_id=None):
        return cls.objects.create(
            user=user,
            action='2fa_verification',
            ip_address=request.META.get('REMOTE_ADDR'),
            device_id=device_id,
            success=success
        )
```

Slide 13: Emergency Access Protocol

The emergency access protocol provides a secure mechanism for authorized personnel to temporarily bypass 2FA requirements in critical situations while maintaining an audit trail of such actions.

```python
# views.py
from django.contrib.admin.views.decorators import staff_member_required
from django.utils import timezone
import uuid

class EmergencyAccess:
    def __init__(self, user):
        self.user = user
        self.token = str(uuid.uuid4())
        self.expiry = timezone.now() + timezone.timedelta(hours=1)

    @classmethod
    @staff_member_required
    def grant_emergency_access(cls, request, user_id):
        cache_key = f'emergency_access_{user_id}'
        emergency_access = cls(user_id)
        cache.set(cache_key, {
            'token': emergency_access.token,
            'expiry': emergency_access.expiry,
            'granted_by': request.user.id
        }, timeout=3600)
        
        SecurityAuditLog.objects.create(
            user_id=user_id,
            action='emergency_access_granted',
            ip_address=request.META.get('REMOTE_ADDR'),
            success=True
        )
        return emergency_access.token
```

Slide 14: Integration Testing Suite

A comprehensive testing suite ensures the reliability and security of the 2FA implementation through unit tests, integration tests, and security vulnerability checks.

```python
# tests.py
from django.test import TestCase, Client
from django.contrib.auth.models import User
from django_otp.plugins.otp_totp.models import TOTPDevice
import pyotp

class TwoFactorAuthTests(TestCase):
    def setUp(self):
        self.client = Client()
        self.user = User.objects.create_user(
            username='testuser',
            password='testpass123'
        )
        self.totp_device = TOTPDevice.objects.create(
            user=self.user,
            name='default',
            confirmed=True
        )
        
    def test_2fa_login_flow(self):
        # First step: password authentication
        response = self.client.post('/login/', {
            'username': 'testuser',
            'password': 'testpass123'
        })
        self.assertEqual(response.status_code, 302)
        
        # Second step: 2FA verification
        totp = pyotp.TOTP(self.totp_device.bin_key)
        response = self.client.post('/verify/', {
            'otp_token': totp.now()
        })
        self.assertEqual(response.status_code, 302)
        self.assertTrue('_auth_user_id' in self.client.session)
```

Slide 15: Additional Resources

1.  [https://arxiv.org/abs/2111.09922](https://arxiv.org/abs/2111.09922) - "A Comprehensive Analysis of Two-Factor Authentication Schemes in Modern Web Applications"
2.  [https://arxiv.org/abs/2103.08364](https://arxiv.org/abs/2103.08364) - "Security Analysis of Time-Based One-Time Passwords in Authentication Systems"
3.  [https://arxiv.org/abs/2008.00959](https://arxiv.org/abs/2008.00959) - "Practical Security Analysis of Two-Factor Authentication Implementations"
4.  [https://arxiv.org/abs/1904.04720](https://arxiv.org/abs/1904.04720) - "A Study on the Implementation and Security of Modern Two-Factor Authentication Methods"
5.  [https://arxiv.org/abs/2201.09120](https://arxiv.org/abs/2201.09120) - "Securing Web Applications with Multi-Factor Authentication: Best Practices and Implementation Guidelines"

