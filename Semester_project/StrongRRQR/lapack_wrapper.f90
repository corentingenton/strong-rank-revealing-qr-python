
subroutine py_dgeqp3rk(m, n, nrhs, a, lda, jpvt, tau, b, ldb, work, lwork, info)
    implicit none
    integer, intent(in) :: m, n, nrhs, lda, ldb, lwork
    real(8), intent(inout) :: a(lda,*), b(ldb,*), tau(*), work(*)
    integer, intent(inout) :: jpvt(*), info

    call dgeqp3rk(m, n, nrhs, a, lda, jpvt, tau, b, ldb, work, lwork, info)
end subroutine py_dgeqp3rk
