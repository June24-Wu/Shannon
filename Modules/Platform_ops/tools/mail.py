import smtplib
import time
from email.mime.text import MIMEText

from config.config import config


def send_mail(subject, msg, receivers: list):
    """
    :param subject: Title of email
    :param msg: Content of email.
    :param receivers: Who will receive the mail.
    :return:
    """

    mail_host = config.MAIL_HOST
    mail_user = config.MAIL_USER
    mail_pass = config.MAIL_PWD
    mail_sender = config.MAIL_SENDER

    message = MIMEText(msg, 'plain', 'utf-8')
    message['Subject'] = f'{subject}'
    message['From'] = mail_sender
    message['To'] = ','.join(receivers)

    for i in range(3):
        try:
            smtp_obj = smtplib.SMTP()
            smtp_obj.connect(mail_host, 25)
            smtp_obj.login(mail_user, mail_pass)
            smtp_obj.sendmail(
                mail_sender, receivers, message.as_string())
            smtp_obj.quit()
        except smtplib.SMTPException:
            time.sleep(2)
        else:
            break
