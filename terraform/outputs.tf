output "application_url" {
  value = "http://${aws_elastic_beanstalk_environment.env.cname}"
}