! file: dgeqp3rk_wrapper.f90
subroutine dgeqp3rk_wrapper(m, n, a, lda, k, jpvt, tau, work, lwork, info)
    implicit none
    ! Input parameters
    integer, intent(in) :: m, n, lda, k, lwork
    ! Input/Output arrays
    double precision, intent(inout), dimension(lda,*) :: a
    integer, intent(inout), dimension(*) :: jpvt
    ! Output arrays
    double precision, intent(out), dimension(*) :: tau
    double precision, intent(out), dimension(*) :: work
    integer, intent(out) :: info

    external dgeqp3rk

    call dgeqp3rk(m, n, a, lda, k, jpvt, tau, work, lwork, info)
end subroutine dgeqp3rk_wrapper