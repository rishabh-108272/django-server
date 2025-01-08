from django.db import models

class RecentScan(models.Model):
    name = models.CharField(max_length=255)
    scan_date = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.name


class FeaturedCategory(models.Model):
    name = models.CharField(max_length=255)
    description = models.TextField()

    def __str__(self):
        return self.name