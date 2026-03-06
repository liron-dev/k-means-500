variable "aws_region" {
  description = "AWS Region for deployment"
  default     = "us-east-1"
}

variable "app_name" {
  description = "Name of the Beanstalk Application"
  default     = "portfolio-optimizer"
}

variable "env_name" {
  description = "Name of the Beanstalk Environment"
  default     = "portfolio-optimizer-prod"
}