import logging
import smtplib
from email.message import EmailMessage

log = logging.getLogger(__name__)


class EmailNotifier:
    def __init__(self, sender, password):
        self.sender = sender
        self.password = password

    def send_email(self, to, subject, body):
        msg = EmailMessage()
        msg.set_content(body)
        msg["Subject"] = subject
        msg["From"] = self.sender
        msg["To"] = to

        # Send the email
        try:
            with smtplib.SMTP("smtp.gmail.com", 587) as server:
                server.starttls()  # Upgrade the connection to encrypted SSL/TLS
                server.login(self.sender, self.password)
                server.send_message(msg)
                log.info("Email sent successfully!")
        except Exception as e:
            raise Exception(f"Failed to send email: {e}")


def notify_on_termination(to, sender, password):
    notifier = EmailNotifier(sender, password)
    notifier.send_email(to, "Computation finished", "Computation finished")
