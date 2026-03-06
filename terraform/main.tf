provider "aws" {
  region = var.aws_region
}

# 1. Archive the app (Automatically includes prices.csv since we updated .dockerignore)
data "archive_file" "app_zip" {
  type        = "zip"
  source_dir  = "${path.module}/../app"
  output_path = "${path.module}/app_version.zip"
}

# 2. S3 Bucket for App Versions
resource "aws_s3_bucket" "eb_bucket" {
  bucket_prefix = "${var.app_name}-deploy-"
  force_destroy = true
}

resource "aws_s3_object" "app_version" {
  bucket = aws_s3_bucket.eb_bucket.id
  key    = "beanstalk/app_version_${data.archive_file.app_zip.output_md5}.zip"
  source = data.archive_file.app_zip.output_path
}

# 3. IAM Roles
resource "aws_iam_role" "eb_ec2_role" {
  name = "${var.app_name}-ec2-role"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = { Service = "ec2.amazonaws.com" }
    }]
  })
}

resource "aws_iam_role_policy_attachment" "eb_web_tier" {
  role       = aws_iam_role.eb_ec2_role.name
  policy_arn = "arn:aws:iam::aws:policy/AWSElasticBeanstalkWebTier"
}

resource "aws_iam_role_policy_attachment" "eb_docker" {
  role       = aws_iam_role.eb_ec2_role.name
  policy_arn = "arn:aws:iam::aws:policy/AWSElasticBeanstalkMulticontainerDocker"
}

resource "aws_iam_instance_profile" "eb_instance_profile" {
  name = "${var.app_name}-instance-profile"
  role = aws_iam_role.eb_ec2_role.name
}

# 4. Beanstalk App & Env
resource "aws_elastic_beanstalk_application" "app" {
  name = var.app_name
}

resource "aws_elastic_beanstalk_application_version" "app_version" {
  name        = "${var.app_name}-${data.archive_file.app_zip.output_md5}"
  application = aws_elastic_beanstalk_application.app.name
  bucket      = aws_s3_bucket.eb_bucket.id
  key         = aws_s3_object.app_version.key
}

resource "aws_elastic_beanstalk_environment" "env" {
  name                = var.env_name
  application         = aws_elastic_beanstalk_application.app.name
  solution_stack_name = "64bit Amazon Linux 2023 v4.10.0 running Docker"
  version_label       = aws_elastic_beanstalk_application_version.app_version.name

  setting {
    namespace = "aws:autoscaling:launchconfiguration"
    name      = "IamInstanceProfile"
    value     = aws_iam_instance_profile.eb_instance_profile.name
  }

  setting {
    namespace = "aws:elasticbeanstalk:environment"
    name      = "EnvironmentType"
    value     = "SingleInstance"
  }

  # UPGRADED: t3.medium provides 4GB RAM, necessary for the K-Means clustering of 500 stocks
  setting {
    namespace = "aws:autoscaling:launchconfiguration"
    name      = "InstanceType"
    value     = "t3.medium"
  }

  # EXTENDED: ML pipeline is "Deep Research" - it needs more than the default 10 min
  setting {
    namespace = "aws:elasticbeanstalk:command"
    name      = "Timeout"
    value     = "1800"
  }
}